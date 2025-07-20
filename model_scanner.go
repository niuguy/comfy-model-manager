package main

import (
	"crypto/md5"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// ModelScanner handles checking for existing models
type ModelScanner struct {
	config *Config
}

// NewModelScanner creates a new model scanner
func NewModelScanner(config *Config) *ModelScanner {
	return &ModelScanner{config: config}
}

// ScanModels checks which models from the list are present locally
func (s *ModelScanner) ScanModels(models []Model) ([]Model, []Model, error) {
	var present, missing []Model

	for _, model := range models {
		exists, err := s.checkModelExists(model)
		if err != nil {
			return nil, nil, fmt.Errorf("error checking model %s: %w", model.Name, err)
		}

		model.IsPresent = exists
		if exists {
			present = append(present, model)
		} else {
			missing = append(missing, model)
		}
	}

	return present, missing, nil
}

// checkModelExists checks if a model file exists locally
func (s *ModelScanner) checkModelExists(model Model) (bool, error) {
	// First check the exact path
	if fileExists(model.LocalPath) {
		return true, nil
	}

	// Check without extension
	baseNameWithoutExt := strings.TrimSuffix(model.Name, filepath.Ext(model.Name))
	dirPath := filepath.Dir(model.LocalPath)

	// Common model extensions
	extensions := []string{
		".safetensors", ".ckpt", ".pt", ".pth", ".bin",
		".yaml", ".json", // for configs
	}

	// Try different extensions
	for _, ext := range extensions {
		testPath := filepath.Join(dirPath, baseNameWithoutExt+ext)
		if fileExists(testPath) {
			model.LocalPath = testPath
			return true, nil
		}
	}

	// Check if it's a directory (some models are directories)
	if dirExists(model.LocalPath) {
		return true, nil
	}

	return false, nil
}

// CalculateModelHash calculates the hash of a model file
func (s *ModelScanner) CalculateModelHash(path string, hashType string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()

	var hasher io.Writer
	switch strings.ToLower(hashType) {
	case "md5":
		h := md5.New()
		hasher = h
		if _, err := io.Copy(hasher, file); err != nil {
			return "", err
		}
		return hex.EncodeToString(h.Sum(nil)), nil
	case "sha256":
		h := sha256.New()
		hasher = h
		if _, err := io.Copy(hasher, file); err != nil {
			return "", err
		}
		return hex.EncodeToString(h.Sum(nil)), nil
	default:
		// For large files, calculate a quick hash of first and last MB
		return s.calculateQuickHash(file)
	}
}

// calculateQuickHash calculates a quick hash for large files
func (s *ModelScanner) calculateQuickHash(file *os.File) (string, error) {
	hasher := sha256.New()

	// Hash first 1MB
	buffer := make([]byte, 1024*1024)
	n, err := file.Read(buffer)
	if err != nil && err != io.EOF {
		return "", err
	}
	hasher.Write(buffer[:n])

	// Get file size
	stat, err := file.Stat()
	if err != nil {
		return "", err
	}

	// Hash last 1MB if file is large enough
	if stat.Size() > 2*1024*1024 {
		_, err = file.Seek(-1024*1024, io.SeekEnd)
		if err != nil {
			return "", err
		}
		n, err = file.Read(buffer)
		if err != nil && err != io.EOF {
			return "", err
		}
		hasher.Write(buffer[:n])
	}

	return hex.EncodeToString(hasher.Sum(nil)), nil
}

// ScanDirectory scans a directory for all model files
func (s *ModelScanner) ScanDirectory(modelType ModelType) ([]Model, error) {
	dir, exists := s.config.ModelDirs[string(modelType)]
	if !exists {
		return nil, fmt.Errorf("unknown model type: %s", modelType)
	}

	fullPath := filepath.Join(s.config.ComfyUIPath, dir)

	var models []Model

	err := filepath.Walk(fullPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Skip files we can't access
		}

		if info.IsDir() {
			return nil
		}

		// Check if it's a model file
		ext := strings.ToLower(filepath.Ext(path))
		modelExts := map[string]bool{
			".safetensors": true,
			".ckpt":        true,
			".pt":          true,
			".pth":         true,
			".bin":         true,
		}

		if modelExts[ext] {
			relPath, _ := filepath.Rel(fullPath, path)
			models = append(models, Model{
				Name:      relPath,
				Type:      modelType,
				LocalPath: path,
				Size:      info.Size(),
				IsPresent: true,
			})
		}

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("error scanning directory: %w", err)
	}

	return models, nil
}

// GetModelInfo retrieves detailed information about a local model
func (s *ModelScanner) GetModelInfo(model Model) (Model, error) {
	info, err := os.Stat(model.LocalPath)
	if err != nil {
		return model, err
	}

	model.Size = info.Size()

	// Calculate hash for small files only
	if model.Size < 100*1024*1024 { // 100MB
		hash, err := s.CalculateModelHash(model.LocalPath, "sha256")
		if err == nil {
			model.Hash = hash
		}
	}

	return model, nil
}

// fileExists checks if a file exists
func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

// dirExists checks if a directory exists
func dirExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}
