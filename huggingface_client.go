package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// HuggingFaceClient handles searching and downloading from HuggingFace
type HuggingFaceClient struct {
	token      string
	httpClient *http.Client
}

// HFSearchResponse represents the HuggingFace search API response
type HFSearchResponse []HFModel

// HFModel represents a model on HuggingFace
type HFModel struct {
	ID           string   `json:"id"`
	ModelID      string   `json:"modelId"`
	Author       string   `json:"author"`
	SHA          string   `json:"sha"`
	LastModified string   `json:"lastModified"`
	Private      bool     `json:"private"`
	Tags         []string `json:"tags"`
	Pipeline     string   `json:"pipeline_tag"`
	LibraryName  string   `json:"library_name"`
}

// HFRepoFile represents a file in a HuggingFace repository
type HFRepoFile struct {
	RFilename string `json:"rfilename"`
	Size      int64  `json:"size"`
	BlobID    string `json:"blobId"`
	LFS       *struct {
		Size        int64  `json:"size"`
		SHA256      string `json:"sha256"`
		PointerSize int    `json:"pointerSize"`
	} `json:"lfs,omitempty"`
}

// NewHuggingFaceClient creates a new HuggingFace client
func NewHuggingFaceClient(token string) *HuggingFaceClient {
	return &HuggingFaceClient{
		token: token,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// SearchModels searches for models on HuggingFace
func (h *HuggingFaceClient) SearchModels(query string, modelType ModelType) ([]SearchResult, error) {
	// Map ComfyUI model types to HF tags/filters
	hfTags := h.getHFTags(modelType)

	searchURL := "https://huggingface.co/api/models"
	params := url.Values{}
	params.Add("search", query)
	params.Add("limit", "10")
	params.Add("full", "true")

	for _, tag := range hfTags {
		params.Add("filter", tag)
	}

	fullURL := fmt.Sprintf("%s?%s", searchURL, params.Encode())

	req, err := http.NewRequest("GET", fullURL, nil)
	if err != nil {
		return nil, err
	}

	if h.token != "" {
		req.Header.Set("Authorization", "Bearer "+h.token)
	}

	resp, err := h.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HF API error: %s - %s", resp.Status, string(body))
	}

	var hfModels HFSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&hfModels); err != nil {
		return nil, err
	}

	// Convert to SearchResults
	results := []SearchResult{}
	for _, model := range hfModels {
		result, err := h.getModelFiles(model, modelType)
		if err != nil {
			continue // Skip models we can't get files for
		}
		results = append(results, result...)
	}

	return results, nil
}

// getModelFiles gets the downloadable files for a model
func (h *HuggingFaceClient) getModelFiles(model HFModel, modelType ModelType) ([]SearchResult, error) {
	filesURL := fmt.Sprintf("https://huggingface.co/api/models/%s/tree/main", model.ID)

	req, err := http.NewRequest("GET", filesURL, nil)
	if err != nil {
		return nil, err
	}

	if h.token != "" {
		req.Header.Set("Authorization", "Bearer "+h.token)
	}

	resp, err := h.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get model files: %s", resp.Status)
	}

	var files []HFRepoFile
	if err := json.NewDecoder(resp.Body).Decode(&files); err != nil {
		return nil, err
	}

	results := []SearchResult{}
	for _, file := range files {
		if h.isModelFile(file.RFilename, modelType) {
			result := SearchResult{
				Name:        file.RFilename,
				Source:      "huggingface",
				DownloadURL: fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", model.ID, file.RFilename),
				Size:        file.Size,
				ModelType:   modelType,
			}

			if file.LFS != nil {
				result.Size = file.LFS.Size
				result.Hash = file.LFS.SHA256
			}

			results = append(results, result)
		}
	}

	return results, nil
}

// getHFTags returns HuggingFace tags for a model type
func (h *HuggingFaceClient) getHFTags(modelType ModelType) []string {
	switch modelType {
	case ModelTypeCheckpoint:
		return []string{"stable-diffusion", "text-to-image"}
	case ModelTypeLora:
		return []string{"lora", "stable-diffusion"}
	case ModelTypeVAE:
		return []string{"vae", "stable-diffusion"}
	case ModelTypeControlNet:
		return []string{"controlnet", "stable-diffusion"}
	case ModelTypeUpscale:
		return []string{"super-resolution", "image-enhancement"}
	case ModelTypeClipVision:
		return []string{"clip", "vision"}
	default:
		return []string{}
	}
}

// isModelFile checks if a filename is likely a model file
func (h *HuggingFaceClient) isModelFile(filename string, modelType ModelType) bool {
	lower := strings.ToLower(filename)

	// Check common model file extensions
	modelExts := []string{".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
	hasModelExt := false
	for _, ext := range modelExts {
		if strings.HasSuffix(lower, ext) {
			hasModelExt = true
			break
		}
	}

	if !hasModelExt {
		return false
	}

	// Filter based on model type
	switch modelType {
	case ModelTypeVAE:
		return strings.Contains(lower, "vae")
	case ModelTypeLora:
		return strings.Contains(lower, "lora") || !strings.Contains(lower, "vae")
	case ModelTypeCheckpoint:
		// Avoid VAE and LoRA files
		return !strings.Contains(lower, "vae") && !strings.Contains(lower, "lora")
	default:
		return true
	}
}

// DownloadFile downloads a file from HuggingFace
func (h *HuggingFaceClient) DownloadFile(downloadURL, destPath string, onProgress func(downloaded, total int64)) error {
	req, err := http.NewRequest("GET", downloadURL, nil)
	if err != nil {
		return err
	}

	if h.token != "" {
		req.Header.Set("Authorization", "Bearer "+h.token)
	}

	resp, err := h.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed: %s", resp.Status)
	}

	return downloadFile(resp.Body, destPath, resp.ContentLength, onProgress)
}
