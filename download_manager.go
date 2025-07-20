package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// DownloadManager handles downloading models
type DownloadManager struct {
	config      *Config
	hfClient    *HuggingFaceClient
	civitClient *CivitAIClient
	workers     int
	mu          sync.Mutex
	downloads   map[string]*DownloadProgress
}

// DownloadProgress tracks download progress
type DownloadProgress struct {
	Model      Model
	Downloaded int64
	Total      int64
	StartTime  time.Time
	Error      error
	Completed  bool
}

// DownloadJob represents a download task
type DownloadJob struct {
	Model        Model
	SearchResult SearchResult
}

// NewDownloadManager creates a new download manager
func NewDownloadManager(config *Config) *DownloadManager {
	return &DownloadManager{
		config:      config,
		hfClient:    NewHuggingFaceClient(config.HuggingFaceToken),
		civitClient: NewCivitAIClient(config.CivitAIToken),
		workers:     config.MaxWorkers,
		downloads:   make(map[string]*DownloadProgress),
	}
}

// DownloadModels downloads a list of models
func (d *DownloadManager) DownloadModels(models []Model, searchResults map[string]SearchResult) error {
	jobs := make(chan DownloadJob, len(models))
	errors := make(chan error, len(models))

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < d.workers; i++ {
		wg.Add(1)
		go d.downloadWorker(&wg, jobs, errors)
	}

	// Queue jobs
	for _, model := range models {
		if result, ok := searchResults[model.Name]; ok {
			jobs <- DownloadJob{
				Model:        model,
				SearchResult: result,
			}
		}
	}
	close(jobs)

	// Wait for workers to finish
	go func() {
		wg.Wait()
		close(errors)
	}()

	// Collect errors
	var downloadErrors []error
	for err := range errors {
		if err != nil {
			downloadErrors = append(downloadErrors, err)
		}
	}

	if len(downloadErrors) > 0 {
		return fmt.Errorf("download errors: %v", downloadErrors)
	}

	return nil
}

// downloadWorker processes download jobs
func (d *DownloadManager) downloadWorker(wg *sync.WaitGroup, jobs <-chan DownloadJob, errors chan<- error) {
	defer wg.Done()

	for job := range jobs {
		err := d.downloadModel(job)
		if err != nil {
			errors <- fmt.Errorf("failed to download %s: %w", job.Model.Name, err)
		} else {
			errors <- nil
		}
	}
}

// downloadModel downloads a single model with retry logic
func (d *DownloadManager) downloadModel(job DownloadJob) error {
	progress := &DownloadProgress{
		Model:     job.Model,
		StartTime: time.Now(),
	}

	d.mu.Lock()
	d.downloads[job.Model.Name] = progress
	d.mu.Unlock()

	// Ensure directory exists
	dir := filepath.Dir(job.Model.LocalPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		progress.Error = err
		return err
	}

	// Download with retries
	var lastErr error
	for attempt := 0; attempt < d.config.RetryAttempts; attempt++ {
		if attempt > 0 {
			fmt.Printf("Retrying download for %s (attempt %d/%d)\n",
				job.Model.Name, attempt+1, d.config.RetryAttempts)
			time.Sleep(time.Second * time.Duration(attempt*2)) // Exponential backoff
		}

		err := d.performDownload(job, progress)
		if err == nil {
			progress.Completed = true
			return nil
		}

		lastErr = err
		progress.Error = err

		// Don't retry on certain errors
		if isUnrecoverableError(err) {
			break
		}
	}

	return lastErr
}

// performDownload performs the actual download
func (d *DownloadManager) performDownload(job DownloadJob, progress *DownloadProgress) error {
	tempPath := job.Model.LocalPath + ".tmp"

	// Check if we can resume a partial download
	var resumeFrom int64
	if info, err := os.Stat(tempPath); err == nil {
		resumeFrom = info.Size()
		progress.Downloaded = resumeFrom
	}

	// Progress callback
	onProgress := func(downloaded, total int64) {
		d.mu.Lock()
		progress.Downloaded = downloaded + resumeFrom
		progress.Total = total
		d.mu.Unlock()

		// Print progress
		if total > 0 {
			percent := float64(progress.Downloaded) / float64(total) * 100
			speed := calculateSpeed(progress.Downloaded-resumeFrom, time.Since(progress.StartTime))
			fmt.Printf("\r%s: %.1f%% (%.2f MB/s)", job.Model.Name, percent, speed)
		}
	}

	// Download based on source
	var err error
	switch job.SearchResult.Source {
	case "huggingface":
		err = d.hfClient.DownloadFile(job.SearchResult.DownloadURL, tempPath, onProgress)
	case "civitai":
		err = d.civitClient.DownloadFile(job.SearchResult.DownloadURL, tempPath, onProgress)
	default:
		err = fmt.Errorf("unknown source: %s", job.SearchResult.Source)
	}

	if err != nil {
		return err
	}

	fmt.Println() // New line after progress

	// Move temp file to final location
	if err := os.Rename(tempPath, job.Model.LocalPath); err != nil {
		return fmt.Errorf("failed to move downloaded file: %w", err)
	}

	return nil
}

// GetProgress returns the current download progress
func (d *DownloadManager) GetProgress() map[string]*DownloadProgress {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Create a copy to avoid race conditions
	progressCopy := make(map[string]*DownloadProgress)
	for k, v := range d.downloads {
		progressCopy[k] = v
	}

	return progressCopy
}

// downloadFile is a helper function to download a file with progress
func downloadFile(reader io.Reader, destPath string, totalSize int64, onProgress func(downloaded, total int64)) error {
	// Create temp file
	tempPath := destPath + ".tmp"

	flags := os.O_CREATE | os.O_WRONLY
	resumeFrom := int64(0)

	// Check if we can resume
	if info, err := os.Stat(tempPath); err == nil {
		resumeFrom = info.Size()
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}

	file, err := os.OpenFile(tempPath, flags, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	// If resuming, seek to the end
	if resumeFrom > 0 {
		if _, err := file.Seek(0, io.SeekEnd); err != nil {
			return err
		}
	}

	// Download with progress tracking
	buf := make([]byte, 1024*1024) // 1MB buffer
	downloaded := resumeFrom

	for {
		n, err := reader.Read(buf)
		if n > 0 {
			if _, err := file.Write(buf[:n]); err != nil {
				return err
			}
			downloaded += int64(n)
			if onProgress != nil {
				onProgress(downloaded, totalSize)
			}
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}

	// Close file before renaming
	file.Close()

	// Move to final location
	if err := os.Rename(tempPath, destPath); err != nil {
		return fmt.Errorf("failed to move temp file: %w", err)
	}

	return nil
}

// calculateSpeed calculates download speed in MB/s
func calculateSpeed(bytes int64, duration time.Duration) float64 {
	if duration.Seconds() == 0 {
		return 0
	}
	return float64(bytes) / (1024 * 1024) / duration.Seconds()
}

// isUnrecoverableError checks if an error should not be retried
func isUnrecoverableError(err error) bool {
	// Add checks for specific error types that shouldn't be retried
	errStr := err.Error()
	unrecoverableErrors := []string{
		"404",
		"403",
		"401",
		"not found",
		"forbidden",
		"unauthorized",
	}

	for _, e := range unrecoverableErrors {
		if strings.Contains(strings.ToLower(errStr), e) {
			return true
		}
	}

	return false
}
