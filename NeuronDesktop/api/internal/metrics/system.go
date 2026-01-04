package metrics

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/disk"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/net"
)

/* SystemMetrics represents current system metrics */
type SystemMetrics struct {
	Timestamp time.Time      `json:"timestamp"`
	CPU       CPUMetrics     `json:"cpu"`
	Memory    MemoryMetrics  `json:"memory"`
	Disk      DiskMetrics    `json:"disk"`
	Network   NetworkMetrics `json:"network"`
	Process   ProcessMetrics `json:"process"`
	GPU       GPUMetrics     `json:"gpu,omitempty"`
}

/* CPUMetrics contains CPU usage information */
type CPUMetrics struct {
	UsagePercent float64   `json:"usage_percent"`
	UsagePerCore []float64 `json:"usage_per_core,omitempty"`
	Count        int       `json:"count"`
	Frequency    float64   `json:"frequency,omitempty"`
	Temperature  float64   `json:"temperature,omitempty"`
}

/* MemoryMetrics contains memory usage information */
type MemoryMetrics struct {
	Total       uint64  `json:"total"`
	Used        uint64  `json:"used"`
	Available   uint64  `json:"available"`
	Free        uint64  `json:"free"`
	UsedPercent float64 `json:"used_percent"`
	Cached      uint64  `json:"cached,omitempty"`
	Buffers     uint64  `json:"buffers,omitempty"`
}

/* DiskMetrics contains disk usage information */
type DiskMetrics struct {
	Total       uint64  `json:"total"`
	Used        uint64  `json:"used"`
	Free        uint64  `json:"free"`
	UsedPercent float64 `json:"used_percent"`
	ReadBytes   uint64  `json:"read_bytes,omitempty"`
	WriteBytes  uint64  `json:"write_bytes,omitempty"`
	ReadCount   uint64  `json:"read_count,omitempty"`
	WriteCount  uint64  `json:"write_count,omitempty"`
}

/* NetworkMetrics contains network usage information */
type NetworkMetrics struct {
	BytesSent     uint64  `json:"bytes_sent"`
	BytesRecv     uint64  `json:"bytes_recv"`
	PacketsSent   uint64  `json:"packets_sent"`
	PacketsRecv   uint64  `json:"packets_recv"`
	BytesSentRate float64 `json:"bytes_sent_rate,omitempty"`
	BytesRecvRate float64 `json:"bytes_recv_rate,omitempty"`
}

/* ProcessMetrics contains process information */
type ProcessMetrics struct {
	Count      int    `json:"count"`
	GoRoutines int    `json:"go_routines"`
	HeapAlloc  uint64 `json:"heap_alloc"`
	HeapSys    uint64 `json:"heap_sys"`
	HeapIdle   uint64 `json:"heap_idle"`
	HeapInuse  uint64 `json:"heap_inuse"`
}

/* GPUMetrics contains GPU usage information (if available) */
type GPUMetrics struct {
	Available   bool        `json:"available"`
	Count       int         `json:"count,omitempty"`
	Usage       []float64   `json:"usage,omitempty"`
	Memory      []GPUMemory `json:"memory,omitempty"`
	Temperature []float64   `json:"temperature,omitempty"`
}

/* GPUMemory contains GPU memory information */
type GPUMemory struct {
	Total uint64 `json:"total"`
	Used  uint64 `json:"used"`
	Free  uint64 `json:"free"`
}

var (
	lastNetworkStats *net.IOCountersStat
	lastNetworkTime  time.Time
)

/* CollectSystemMetrics collects current system metrics */
func CollectSystemMetrics(ctx context.Context) (*SystemMetrics, error) {
	metrics := &SystemMetrics{
		Timestamp: time.Now(),
	}

	cpuPercent, err := cpu.PercentWithContext(ctx, time.Second, false)
	if err == nil && len(cpuPercent) > 0 {
		metrics.CPU.UsagePercent = cpuPercent[0]
	}

	cpuPercentAll, err := cpu.PercentWithContext(ctx, time.Second, true)
	if err == nil {
		metrics.CPU.UsagePerCore = cpuPercentAll
	}

	cpuCount, err := cpu.Counts(true)
	if err == nil {
		metrics.CPU.Count = cpuCount
	}

	cpuInfo, err := cpu.InfoWithContext(ctx)
	if err == nil && len(cpuInfo) > 0 {
		metrics.CPU.Frequency = cpuInfo[0].Mhz
	}

	memStat, err := mem.VirtualMemoryWithContext(ctx)
	if err == nil {
		metrics.Memory.Total = memStat.Total
		metrics.Memory.Used = memStat.Used
		metrics.Memory.Available = memStat.Available
		metrics.Memory.Free = memStat.Free
		metrics.Memory.UsedPercent = memStat.UsedPercent
		metrics.Memory.Cached = memStat.Cached
		metrics.Memory.Buffers = memStat.Buffers
	}

	diskStat, err := disk.UsageWithContext(ctx, "/")
	if err == nil {
		metrics.Disk.Total = diskStat.Total
		metrics.Disk.Used = diskStat.Used
		metrics.Disk.Free = diskStat.Free
		metrics.Disk.UsedPercent = diskStat.UsedPercent
	}

	diskIO, err := disk.IOCountersWithContext(ctx)
	if err == nil {
		var totalRead, totalWrite uint64
		var totalReadCount, totalWriteCount uint64
		for _, io := range diskIO {
			totalRead += io.ReadBytes
			totalWrite += io.WriteBytes
			totalReadCount += io.ReadCount
			totalWriteCount += io.WriteCount
		}
		metrics.Disk.ReadBytes = totalRead
		metrics.Disk.WriteBytes = totalWrite
		metrics.Disk.ReadCount = totalReadCount
		metrics.Disk.WriteCount = totalWriteCount
	}

	netIO, err := net.IOCountersWithContext(ctx, false)
	if err == nil && len(netIO) > 0 {
		stats := netIO[0]
		metrics.Network.BytesSent = stats.BytesSent
		metrics.Network.BytesRecv = stats.BytesRecv
		metrics.Network.PacketsSent = stats.PacketsSent
		metrics.Network.PacketsRecv = stats.PacketsRecv

		if lastNetworkStats != nil && !lastNetworkTime.IsZero() {
			elapsed := time.Since(lastNetworkTime).Seconds()
			if elapsed > 0 {
				metrics.Network.BytesSentRate = float64(stats.BytesSent-lastNetworkStats.BytesSent) / elapsed
				metrics.Network.BytesRecvRate = float64(stats.BytesRecv-lastNetworkStats.BytesRecv) / elapsed
			}
		}
		lastNetworkStats = &stats
		lastNetworkTime = time.Now()
	}

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	metrics.Process.GoRoutines = runtime.NumGoroutine()
	metrics.Process.HeapAlloc = m.HeapAlloc
	metrics.Process.HeapSys = m.HeapSys
	metrics.Process.HeapIdle = m.HeapIdle
	metrics.Process.HeapInuse = m.HeapInuse

	gpuMetrics, err := detectGPU(ctx)
	if err == nil {
		metrics.GPU = *gpuMetrics
	} else {
		metrics.GPU.Available = false
	}

	return metrics, nil
}

/* detectGPU detects and collects GPU metrics */
func detectGPU(ctx context.Context) (*GPUMetrics, error) {
	metrics := &GPUMetrics{
		Available: false,
	}

	if nvidiaMetrics, err := detectNVIDIAGPU(ctx); err == nil {
		metrics = nvidiaMetrics
		return metrics, nil
	}

	if amdMetrics, err := detectAMDGPU(ctx); err == nil {
		metrics = amdMetrics
		return metrics, nil
	}

	if runtime.GOOS == "darwin" {
		if metalMetrics, err := detectAppleMetalGPU(ctx); err == nil {
			metrics = metalMetrics
			return metrics, nil
		}
	}

	return metrics, fmt.Errorf("no GPU detected")
}

/* detectNVIDIAGPU detects NVIDIA GPUs using nvidia-smi */
func detectNVIDIAGPU(ctx context.Context) (*GPUMetrics, error) {
	cmd := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi not available: %w", err)
	}

	metrics := &GPUMetrics{
		Available: true,
		Usage:     []float64{},
		Memory:    []GPUMemory{},
		Temperature: []float64{},
	}

	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	metrics.Count = len(lines)

	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}

		parts := strings.Split(line, ", ")
		if len(parts) < 6 {
			continue
		}

		if usage, err := strconv.ParseFloat(strings.TrimSpace(parts[2]), 64); err == nil {
			metrics.Usage = append(metrics.Usage, usage)
		}

		memUsed, _ := strconv.ParseUint(strings.TrimSpace(parts[3]), 10, 64)
		memTotal, _ := strconv.ParseUint(strings.TrimSpace(parts[4]), 10, 64)
		metrics.Memory = append(metrics.Memory, GPUMemory{
			Used:  memUsed * 1024 * 1024, // Convert MB to bytes
			Total: memTotal * 1024 * 1024,
			Free:  (memTotal - memUsed) * 1024 * 1024,
		})

		if temp, err := strconv.ParseFloat(strings.TrimSpace(parts[5]), 64); err == nil {
			metrics.Temperature = append(metrics.Temperature, temp)
		}
	}

	if metrics.Count == 0 {
		return nil, fmt.Errorf("no NVIDIA GPUs found")
	}

	return metrics, nil
}

/* detectAMDGPU detects AMD GPUs using rocm-smi */
func detectAMDGPU(ctx context.Context) (*GPUMetrics, error) {
	cmd := exec.CommandContext(ctx, "rocm-smi", "--showid", "--showtemp", "--showmemuse", "--showmeminfo", "vram", "--csv")
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("rocm-smi not available: %w", err)
	}

	metrics := &GPUMetrics{
		Available:   true,
		Usage:       []float64{},
		Memory:      []GPUMemory{},
		Temperature: []float64{},
	}

	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) < 2 {
		return nil, fmt.Errorf("invalid rocm-smi output")
	}

	for i := 1; i < len(lines); i++ {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}

		parts := strings.Split(line, ",")
		if len(parts) < 5 {
			continue
		}

		memParts := strings.Split(strings.TrimSpace(parts[4]), "/")
		if len(memParts) == 2 {
			memUsed, _ := strconv.ParseUint(strings.TrimSpace(memParts[0]), 10, 64)
			memTotal, _ := strconv.ParseUint(strings.TrimSpace(memParts[1]), 10, 64)
			metrics.Memory = append(metrics.Memory, GPUMemory{
				Used:  memUsed * 1024 * 1024, // Convert MB to bytes
				Total: memTotal * 1024 * 1024,
				Free:  (memTotal - memUsed) * 1024 * 1024,
			})
		}

		if temp, err := strconv.ParseFloat(strings.TrimSpace(parts[2]), 64); err == nil {
			metrics.Temperature = append(metrics.Temperature, temp)
		}

		metrics.Usage = append(metrics.Usage, 0.0)
	}

	metrics.Count = len(metrics.Memory)
	if metrics.Count == 0 {
		return nil, fmt.Errorf("no AMD GPUs found")
	}

	return metrics, nil
}

/* detectAppleMetalGPU detects Apple Metal GPUs using system_profiler */
func detectAppleMetalGPU(ctx context.Context) (*GPUMetrics, error) {
	cmd := exec.CommandContext(ctx, "system_profiler", "SPDisplaysDataType", "-json")
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("system_profiler not available: %w", err)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse system_profiler output: %w", err)
	}

	metrics := &GPUMetrics{
		Available:   true,
		Usage:       []float64{},
		Memory:      []GPUMemory{},
		Temperature: []float64{},
	}

	if displays, ok := result["SPDisplaysData"].([]interface{}); ok {
		for _, display := range displays {
			if displayMap, ok := display.(map[string]interface{}); ok {
				if gpuName, ok := displayMap["sppci_model"].(string); ok && strings.Contains(strings.ToLower(gpuName), "metal") {
					metrics.Count++

					metrics.Memory = append(metrics.Memory, GPUMemory{
						Total: 0, // Not available via system_profiler
						Used:  0,
						Free:  0,
					})
					metrics.Usage = append(metrics.Usage, 0.0) // Not available
					metrics.Temperature = append(metrics.Temperature, 0.0) // Not available
				}
			}
		}
	}

	if metrics.Count == 0 {
		return nil, fmt.Errorf("no Apple Metal GPUs found")
	}

	return metrics, nil
}

/* GetMetricsJSON returns metrics as JSON string */
func GetMetricsJSON(ctx context.Context) (string, error) {
	metrics, err := CollectSystemMetrics(ctx)
	if err != nil {
		return "", err
	}
	data, err := json.Marshal(metrics)
	if err != nil {
		return "", err
	}
	return string(data), nil
}
