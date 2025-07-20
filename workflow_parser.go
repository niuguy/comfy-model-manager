package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// WorkflowParser handles parsing ComfyUI workflows
type WorkflowParser struct {
	config *Config
}

// NewWorkflowParser creates a new workflow parser
func NewWorkflowParser(config *Config) *WorkflowParser {
	return &WorkflowParser{config: config}
}

// ParseWorkflow parses a workflow file and extracts model references
func (p *WorkflowParser) ParseWorkflow(path string) ([]Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read workflow file: %w", err)
	}

	var workflow Workflow
	if err := json.Unmarshal(data, &workflow); err != nil {
		return nil, fmt.Errorf("failed to parse workflow JSON: %w", err)
	}

	models := p.extractModels(workflow)
	return models, nil
}

// extractModels extracts all model references from the workflow
func (p *WorkflowParser) extractModels(workflow Workflow) []Model {
	modelMap := make(map[string]Model)

	for _, node := range workflow {
		switch node.ClassType {
		case "CheckpointLoaderSimple", "CheckpointLoader":
			p.extractCheckpoint(node, modelMap)
		case "LoraLoader", "LoraLoaderModelOnly":
			p.extractLora(node, modelMap)
		case "VAELoader":
			p.extractVAE(node, modelMap)
		case "ControlNetLoader":
			p.extractControlNet(node, modelMap)
		case "CLIPVisionLoader":
			p.extractClipVision(node, modelMap)
		case "UpscaleModelLoader":
			p.extractUpscaleModel(node, modelMap)
		default:
			// Check for embedding references in text fields
			p.extractEmbeddings(node, modelMap)
		}
	}

	// Convert map to slice
	models := make([]Model, 0, len(modelMap))
	for _, model := range modelMap {
		models = append(models, model)
	}

	return models
}

// extractCheckpoint extracts checkpoint model references
func (p *WorkflowParser) extractCheckpoint(node WorkflowNode, modelMap map[string]Model) {
	if ckptName, ok := node.Inputs["ckpt_name"].(string); ok {
		key := fmt.Sprintf("%s:%s", ModelTypeCheckpoint, ckptName)
		modelMap[key] = Model{
			Name:      ckptName,
			Type:      ModelTypeCheckpoint,
			LocalPath: p.config.GetModelPath(ModelTypeCheckpoint, ckptName),
		}
	}
}

// extractLora extracts LoRA model references
func (p *WorkflowParser) extractLora(node WorkflowNode, modelMap map[string]Model) {
	if loraName, ok := node.Inputs["lora_name"].(string); ok {
		key := fmt.Sprintf("%s:%s", ModelTypeLora, loraName)
		modelMap[key] = Model{
			Name:      loraName,
			Type:      ModelTypeLora,
			LocalPath: p.config.GetModelPath(ModelTypeLora, loraName),
		}
	}
}

// extractVAE extracts VAE model references
func (p *WorkflowParser) extractVAE(node WorkflowNode, modelMap map[string]Model) {
	if vaeName, ok := node.Inputs["vae_name"].(string); ok {
		key := fmt.Sprintf("%s:%s", ModelTypeVAE, vaeName)
		modelMap[key] = Model{
			Name:      vaeName,
			Type:      ModelTypeVAE,
			LocalPath: p.config.GetModelPath(ModelTypeVAE, vaeName),
		}
	}
}

// extractControlNet extracts ControlNet model references
func (p *WorkflowParser) extractControlNet(node WorkflowNode, modelMap map[string]Model) {
	if controlNetName, ok := node.Inputs["control_net_name"].(string); ok {
		key := fmt.Sprintf("%s:%s", ModelTypeControlNet, controlNetName)
		modelMap[key] = Model{
			Name:      controlNetName,
			Type:      ModelTypeControlNet,
			LocalPath: p.config.GetModelPath(ModelTypeControlNet, controlNetName),
		}
	}
}

// extractClipVision extracts CLIP Vision model references
func (p *WorkflowParser) extractClipVision(node WorkflowNode, modelMap map[string]Model) {
	if clipName, ok := node.Inputs["clip_name"].(string); ok {
		key := fmt.Sprintf("%s:%s", ModelTypeClipVision, clipName)
		modelMap[key] = Model{
			Name:      clipName,
			Type:      ModelTypeClipVision,
			LocalPath: p.config.GetModelPath(ModelTypeClipVision, clipName),
		}
	}
}

// extractUpscaleModel extracts upscale model references
func (p *WorkflowParser) extractUpscaleModel(node WorkflowNode, modelMap map[string]Model) {
	if modelName, ok := node.Inputs["model_name"].(string); ok {
		key := fmt.Sprintf("%s:%s", ModelTypeUpscale, modelName)
		modelMap[key] = Model{
			Name:      modelName,
			Type:      ModelTypeUpscale,
			LocalPath: p.config.GetModelPath(ModelTypeUpscale, modelName),
		}
	}
}

// extractEmbeddings extracts embedding references from text fields
func (p *WorkflowParser) extractEmbeddings(node WorkflowNode, modelMap map[string]Model) {
	// Look for embedding syntax in text fields (e.g., "embedding:easynegative")
	for _, input := range node.Inputs {
		if text, ok := input.(string); ok {
			embeddings := p.findEmbeddings(text)
			for _, embedding := range embeddings {
				key := fmt.Sprintf("%s:%s", ModelTypeEmbedding, embedding)
				modelMap[key] = Model{
					Name:      embedding,
					Type:      ModelTypeEmbedding,
					LocalPath: p.config.GetModelPath(ModelTypeEmbedding, embedding),
				}
			}
		}
	}
}

// findEmbeddings finds embedding references in text
func (p *WorkflowParser) findEmbeddings(text string) []string {
	var embeddings []string

	// Look for patterns like "embedding:name" or "(embedding:name:weight)"
	parts := strings.Split(text, "embedding:")
	for i := 1; i < len(parts); i++ {
		// Extract the embedding name
		endIdx := strings.IndexAny(parts[i], " ,():")
		if endIdx == -1 {
			endIdx = len(parts[i])
		}

		if endIdx > 0 {
			embName := parts[i][:endIdx]
			// Add file extension if not present
			if !strings.Contains(embName, ".") {
				embName += ".pt"
			}
			embeddings = append(embeddings, embName)
		}
	}

	return embeddings
}
