package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Config holds the configuration for the model manager
type Config struct {
	ComfyUIPath      string            `json:"comfyui_path"`
	HuggingFaceToken string            `json:"huggingface_token"`
	CivitAIToken     string            `json:"civitai_token"`
	MaxWorkers       int               `json:"max_workers"`
	ModelDirs        map[string]string `json:"model_dirs"`
	DownloadTimeout  time.Duration     `json:"download_timeout"`
	RetryAttempts    int               `json:"retry_attempts"`
}

// ModelType represents different types of models in ComfyUI
type ModelType string

const (
	ModelTypeCheckpoint ModelType = "checkpoints"
	ModelTypeLora       ModelType = "loras"
	ModelTypeVAE        ModelType = "vae"
	ModelTypeEmbedding  ModelType = "embeddings"
	ModelTypeControlNet ModelType = "controlnet"
	ModelTypeUpscale    ModelType = "upscale_models"
	ModelTypeClipVision ModelType = "clip_vision"
)

// Model represents a model referenced in a workflow
type Model struct {
	Name        string    `json:"name"`
	Type        ModelType `json:"type"`
	Hash        string    `json:"hash,omitempty"`
	Source      string    `json:"source,omitempty"`
	DownloadURL string    `json:"download_url,omitempty"`
	LocalPath   string    `json:"local_path,omitempty"`
	Size        int64     `json:"size,omitempty"`
	IsPresent   bool      `json:"is_present"`
}

// WorkflowNode represents a node in the ComfyUI workflow
type WorkflowNode struct {
	ClassType string                 `json:"class_type"`
	Inputs    map[string]interface{} `json:"inputs"`
}

// Workflow represents a ComfyUI workflow
type Workflow map[string]WorkflowNode

// SearchResult represents a model search result from HF or CivitAI
type SearchResult struct {
	Name        string
	Source      string // "huggingface" or "civitai"
	DownloadURL string
	Hash        string
	Size        int64
	ModelType   ModelType
}

// DefaultConfig returns a default configuration
func DefaultConfig() *Config {
	return &Config{
		ComfyUIPath:     "/workspace/ComfyUI",
		MaxWorkers:      3,
		DownloadTimeout: 30 * time.Minute,
		RetryAttempts:   3,
		ModelDirs: map[string]string{
			string(ModelTypeCheckpoint): "models/checkpoints",
			string(ModelTypeLora):       "models/loras",
			string(ModelTypeVAE):        "models/vae",
			string(ModelTypeEmbedding):  "models/embeddings",
			string(ModelTypeControlNet): "models/controlnet",
			string(ModelTypeUpscale):    "models/upscale_models",
			string(ModelTypeClipVision): "models/clip_vision",
		},
	}
}

// LoadConfig loads configuration from a JSON file
func LoadConfig(path string) (*Config, error) {
	config := DefaultConfig()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			// Return default config if file doesn't exist
			return config, nil
		}
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	if err := json.Unmarshal(data, config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	return config, nil
}

// GetModelPath returns the full path for a model
func (c *Config) GetModelPath(modelType ModelType, filename string) string {
	dir, exists := c.ModelDirs[string(modelType)]
	if !exists {
		dir = "models/unknown"
	}
	return filepath.Join(c.ComfyUIPath, dir, filename)
}
