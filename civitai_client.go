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

// CivitAIClient handles searching and downloading from CivitAI
type CivitAIClient struct {
	token      string
	httpClient *http.Client
}

// CivitAISearchResponse represents the CivitAI search API response
type CivitAISearchResponse struct {
	Items    []CivitAIModel `json:"items"`
	Metadata struct {
		TotalItems  int    `json:"totalItems"`
		CurrentPage int    `json:"currentPage"`
		PageSize    int    `json:"pageSize"`
		TotalPages  int    `json:"totalPages"`
		NextPage    string `json:"nextPage"`
	} `json:"metadata"`
}

// CivitAIModel represents a model on CivitAI
type CivitAIModel struct {
	ID            int                   `json:"id"`
	Name          string                `json:"name"`
	Type          string                `json:"type"`
	ModelVersions []CivitAIModelVersion `json:"modelVersions"`
	Creator       struct {
		Username string `json:"username"`
	} `json:"creator"`
}

// CivitAIModelVersion represents a version of a model
type CivitAIModelVersion struct {
	ID          int                `json:"id"`
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Files       []CivitAIModelFile `json:"files"`
	Images      []struct {
		URL string `json:"url"`
	} `json:"images"`
}

// CivitAIModelFile represents a downloadable file
type CivitAIModelFile struct {
	ID               int     `json:"id"`
	Name             string  `json:"name"`
	SizeKB           float64 `json:"sizeKB"`
	Type             string  `json:"type"`
	Format           string  `json:"format"`
	PickleScanResult string  `json:"pickleScanResult"`
	VirusScanResult  string  `json:"virusScanResult"`
	Hashes           struct {
		SHA256 string `json:"SHA256"`
		AutoV1 string `json:"AutoV1"`
		AutoV2 string `json:"AutoV2"`
		CRC32  string `json:"CRC32"`
		BLAKE3 string `json:"BLAKE3"`
	} `json:"hashes"`
	DownloadURL string `json:"downloadUrl"`
}

// NewCivitAIClient creates a new CivitAI client
func NewCivitAIClient(token string) *CivitAIClient {
	return &CivitAIClient{
		token: token,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// SearchModels searches for models on CivitAI
func (c *CivitAIClient) SearchModels(query string, modelType ModelType) ([]SearchResult, error) {
	civitType := c.getCivitAIType(modelType)

	searchURL := "https://civitai.com/api/v1/models"
	params := url.Values{}
	params.Add("query", query)
	params.Add("limit", "20")

	if civitType != "" {
		params.Add("types", civitType)
	}

	fullURL := fmt.Sprintf("%s?%s", searchURL, params.Encode())

	req, err := http.NewRequest("GET", fullURL, nil)
	if err != nil {
		return nil, err
	}

	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("CivitAI API error: %s - %s", resp.Status, string(body))
	}

	var searchResp CivitAISearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return nil, err
	}

	// Convert to SearchResults
	results := []SearchResult{}
	for _, model := range searchResp.Items {
		for _, version := range model.ModelVersions {
			for _, file := range version.Files {
				if c.isValidFile(file) {
					result := SearchResult{
						Name:        file.Name,
						Source:      "civitai",
						DownloadURL: c.getDownloadURL(file),
						Hash:        file.Hashes.SHA256,
						Size:        int64(file.SizeKB * 1024),
						ModelType:   modelType,
					}
					results = append(results, result)
				}
			}
		}
	}

	return results, nil
}

// getCivitAIType converts our model type to CivitAI type
func (c *CivitAIClient) getCivitAIType(modelType ModelType) string {
	switch modelType {
	case ModelTypeCheckpoint:
		return "Checkpoint"
	case ModelTypeLora:
		return "LORA"
	case ModelTypeVAE:
		return "VAE"
	case ModelTypeControlNet:
		return "Controlnet"
	case ModelTypeUpscale:
		return "Upscaler"
	case ModelTypeEmbedding:
		return "TextualInversion"
	default:
		return ""
	}
}

// isValidFile checks if a file is safe to download
func (c *CivitAIClient) isValidFile(file CivitAIModelFile) bool {
	// Check virus scan results
	if file.VirusScanResult != "" && file.VirusScanResult != "Success" {
		return false
	}

	// Check pickle scan for Python files
	if strings.HasSuffix(strings.ToLower(file.Name), ".ckpt") ||
		strings.HasSuffix(strings.ToLower(file.Name), ".pt") {
		if file.PickleScanResult != "" && file.PickleScanResult != "Success" {
			return false
		}
	}

	// Check file format
	validFormats := map[string]bool{
		"SafeTensor":   true,
		"PickleTensor": true,
		"Model":        true,
		"Other":        true,
	}

	return validFormats[file.Format]
}

// getDownloadURL constructs the download URL for a file
func (c *CivitAIClient) getDownloadURL(file CivitAIModelFile) string {
	if file.DownloadURL != "" {
		return file.DownloadURL
	}
	// Fallback URL construction
	return fmt.Sprintf("https://civitai.com/api/download/models/%d", file.ID)
}

// GetModelByHash searches for a model by its hash
func (c *CivitAIClient) GetModelByHash(hash string) (*SearchResult, error) {
	searchURL := fmt.Sprintf("https://civitai.com/api/v1/model-versions/by-hash/%s", hash)

	req, err := http.NewRequest("GET", searchURL, nil)
	if err != nil {
		return nil, err
	}

	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, nil // Model not found
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("CivitAI API error: %s", resp.Status)
	}

	var version CivitAIModelVersion
	if err := json.NewDecoder(resp.Body).Decode(&version); err != nil {
		return nil, err
	}

	// Find the primary file
	for _, file := range version.Files {
		if c.isValidFile(file) && file.Type == "Model" {
			return &SearchResult{
				Name:        file.Name,
				Source:      "civitai",
				DownloadURL: c.getDownloadURL(file),
				Hash:        file.Hashes.SHA256,
				Size:        int64(file.SizeKB * 1024),
			}, nil
		}
	}

	return nil, nil
}

// DownloadFile downloads a file from CivitAI
func (c *CivitAIClient) DownloadFile(downloadURL, destPath string, onProgress func(downloaded, total int64)) error {
	req, err := http.NewRequest("GET", downloadURL, nil)
	if err != nil {
		return err
	}

	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
		// CivitAI might require token as query parameter for downloads
		if !strings.Contains(downloadURL, "token=") {
			sep := "?"
			if strings.Contains(downloadURL, "?") {
				sep = "&"
			}
			downloadURL = fmt.Sprintf("%s%stoken=%s", downloadURL, sep, c.token)
			req, _ = http.NewRequest("GET", downloadURL, nil)
		}
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("download failed: %s - %s", resp.Status, string(body))
	}

	return downloadFile(resp.Body, destPath, resp.ContentLength, onProgress)
}
