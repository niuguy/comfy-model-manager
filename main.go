package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// ModelManager is the main application struct
type ModelManager struct {
	config     *Config
	parser     *WorkflowParser
	scanner    *ModelScanner
	downloader *DownloadManager
}

// NewModelManager creates a new model manager instance
func NewModelManager(configPath string) (*ModelManager, error) {
	config, err := LoadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	return &ModelManager{
		config:     config,
		parser:     NewWorkflowParser(config),
		scanner:    NewModelScanner(config),
		downloader: NewDownloadManager(config),
	}, nil
}

// ProcessWorkflow processes a ComfyUI workflow and downloads missing models
func (m *ModelManager) ProcessWorkflow(workflowPath string) error {
	fmt.Printf("Processing workflow: %s\n", workflowPath)

	// Step 1: Parse workflow
	fmt.Println("\n1. Parsing workflow...")
	models, err := m.parser.ParseWorkflow(workflowPath)
	if err != nil {
		return fmt.Errorf("failed to parse workflow: %w", err)
	}
	fmt.Printf("Found %d model references\n", len(models))

	// Step 2: Scan for missing models
	fmt.Println("\n2. Checking for missing models...")
	present, missing, err := m.scanner.ScanModels(models)
	if err != nil {
		return fmt.Errorf("failed to scan models: %w", err)
	}

	fmt.Printf("Present models: %d\n", len(present))
	fmt.Printf("Missing models: %d\n", len(missing))

	if len(missing) == 0 {
		fmt.Println("\nAll models are present! No downloads needed.")
		return nil
	}

	// Print missing models
	fmt.Println("\nMissing models:")
	for _, model := range missing {
		fmt.Printf("  - %s (%s)\n", model.Name, model.Type)
	}

	// Step 3: Search for missing models
	fmt.Println("\n3. Searching for models...")
	searchResults := m.searchModels(missing)

	// Print search results
	fmt.Printf("\nFound %d models online:\n", len(searchResults))
	for name, result := range searchResults {
		fmt.Printf("  - %s: %s (%.2f MB)\n",
			name, result.Source, float64(result.Size)/(1024*1024))
	}

	// Find models that couldn't be found
	notFound := []Model{}
	for _, model := range missing {
		if _, ok := searchResults[model.Name]; !ok {
			notFound = append(notFound, model)
		}
	}

	if len(notFound) > 0 {
		fmt.Println("\nCould not find these models:")
		for _, model := range notFound {
			fmt.Printf("  - %s (%s)\n", model.Name, model.Type)
		}
	}

	// Step 4: Download missing models
	if len(searchResults) > 0 {
		fmt.Println("\n4. Downloading models...")
		err = m.downloader.DownloadModels(missing, searchResults)
		if err != nil {
			return fmt.Errorf("download failed: %w", err)
		}
		fmt.Println("\nAll downloads completed!")
	}

	return nil
}

// searchModels searches for models on HuggingFace and CivitAI
func (m *ModelManager) searchModels(models []Model) map[string]SearchResult {
	results := make(map[string]SearchResult)
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Search concurrently
	for _, model := range models {
		wg.Add(1)
		go func(model Model) {
			defer wg.Done()

			result := m.searchModel(model)
			if result != nil {
				mu.Lock()
				results[model.Name] = *result
				mu.Unlock()
			}
		}(model)
	}

	wg.Wait()
	return results
}

// searchModel searches for a single model
func (m *ModelManager) searchModel(model Model) *SearchResult {
	// Clean up model name for searching
	searchName := cleanModelName(model.Name)

	// Try HuggingFace first
	if m.config.HuggingFaceToken != "" {
		hfResults, err := m.downloader.hfClient.SearchModels(searchName, model.Type)
		if err == nil && len(hfResults) > 0 {
			// Return the first result
			return &hfResults[0]
		}
	}

	// Try CivitAI
	civitResults, err := m.downloader.civitClient.SearchModels(searchName, model.Type)
	if err == nil && len(civitResults) > 0 {
		// Return the first result
		return &civitResults[0]
	}

	// Try searching by hash if available
	if model.Hash != "" {
		if m.config.CivitAIToken != "" {
			result, err := m.downloader.civitClient.GetModelByHash(model.Hash)
			if err == nil && result != nil {
				result.ModelType = model.Type
				return result
			}
		}
	}

	return nil
}

// cleanModelName cleans up a model name for searching
func cleanModelName(name string) string {
	// Remove file extension
	name = strings.TrimSuffix(name, filepath.Ext(name))

	// Remove common suffixes
	suffixes := []string{"_fp16", "_fp32", "-fp16", "-fp32", "_pruned", "-pruned"}
	for _, suffix := range suffixes {
		name = strings.TrimSuffix(name, suffix)
	}

	return name
}

// ScanAllModels scans all model directories
func (m *ModelManager) ScanAllModels() error {
	fmt.Println("Scanning all model directories...")

	modelTypes := []ModelType{
		ModelTypeCheckpoint,
		ModelTypeLora,
		ModelTypeVAE,
		ModelTypeEmbedding,
		ModelTypeControlNet,
		ModelTypeUpscale,
		ModelTypeClipVision,
	}

	for _, modelType := range modelTypes {
		models, err := m.scanner.ScanDirectory(modelType)
		if err != nil {
			log.Printf("Error scanning %s: %v\n", modelType, err)
			continue
		}

		fmt.Printf("\n%s: %d models\n", modelType, len(models))
		for _, model := range models {
			fmt.Printf("  - %s (%.2f MB)\n",
				model.Name, float64(model.Size)/(1024*1024))
		}
	}

	return nil
}

// SaveConfig saves the current configuration
func (m *ModelManager) SaveConfig(path string) error {
	data, err := json.MarshalIndent(m.config, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

func main() {
	var (
		configPath   = flag.String("config", "config.json", "Configuration file path")
		workflowPath = flag.String("workflow", "", "ComfyUI workflow file to process")
		scanOnly     = flag.Bool("scan", false, "Only scan for models, don't download")
		listModels   = flag.Bool("list", false, "List all installed models")
		genConfig    = flag.Bool("gen-config", false, "Generate default configuration file")
	)

	flag.Parse()

	// Generate config if requested
	if *genConfig {
		config := DefaultConfig()
		if err := saveDefaultConfig(*configPath, config); err != nil {
			log.Fatalf("Failed to generate config: %v", err)
		}
		fmt.Printf("Generated default configuration at: %s\n", *configPath)
		return
	}

	// Create model manager
	manager, err := NewModelManager(*configPath)
	if err != nil {
		log.Fatalf("Failed to initialize: %v", err)
	}

	// List models if requested
	if *listModels {
		if err := manager.ScanAllModels(); err != nil {
			log.Fatalf("Failed to scan models: %v", err)
		}
		return
	}

	// Process workflow
	if *workflowPath != "" {
		if *scanOnly {
			// Just scan and report
			models, err := manager.parser.ParseWorkflow(*workflowPath)
			if err != nil {
				log.Fatalf("Failed to parse workflow: %v", err)
			}

			_, missing, err := manager.scanner.ScanModels(models)
			if err != nil {
				log.Fatalf("Failed to scan models: %v", err)
			}

			if len(missing) == 0 {
				fmt.Println("All models are present!")
			} else {
				fmt.Printf("Missing %d models:\n", len(missing))
				for _, model := range missing {
					fmt.Printf("  - %s (%s)\n", model.Name, model.Type)
				}
			}
		} else {
			// Full processing with downloads
			if err := manager.ProcessWorkflow(*workflowPath); err != nil {
				log.Fatalf("Workflow processing failed: %v", err)
			}
		}
		return
	}

	// No workflow specified
	flag.Usage()
}

// saveDefaultConfig saves a default configuration file
func saveDefaultConfig(path string, config *Config) error {
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}
