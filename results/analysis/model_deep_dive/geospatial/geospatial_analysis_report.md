# Geospatial Error Analysis Report

Generated: 2025-12-30 17:01

## 1. Analysis Parameters

| Item | Value |
|:---|---:|
| **Threshold** | **0.1400** |
| Test accidents | 282,376 |
| Fatal accidents | 2,591 |
| Mortality rate | 0.92% |

## 2. Overall Statistics

| Class | Count | Ratio |
|:---|---:|---:|
| True Positive | 762 | 0.27% |
| False Positive | 2,129 | 0.75% |
| False Negative | 1,829 | 0.65% |
| True Negative | 277,656 | 98.33% |

**Recall: 29.41%** | **Precision: 26.36%**

## 3. Area Analysis

Total areas: 50 clusters

### Worst Performing Areas (Top 10)

| Area ID | Fatal | FN | Recall | Coordinates | Road Type |
|:---:|---:|---:|---:|:---|:---|
| 11 | 47 | 41 | 12.8% | (36.33, 139.15) | 13.0 |
| 18 | 29 | 25 | 13.8% | (34.92, 138.34) | 13.0 |
| 49 | 36 | 31 | 13.9% | (35.14, 138.87) | 13.0 |
| 4 | 117 | 100 | 14.5% | (35.64, 139.73) | 13.0 |
| 46 | 81 | 69 | 14.8% | (35.77, 139.44) | 13.0 |
| 16 | 64 | 53 | 17.2% | (33.49, 130.44) | 13.0 |
| 44 | 52 | 43 | 17.3% | (36.26, 139.70) | 13.0 |
| 3 | 69 | 57 | 17.4% | (34.75, 135.62) | 13.0 |
| 21 | 108 | 89 | 17.6% | (35.86, 139.78) | 13.0 |
| 35 | 92 | 74 | 19.6% | (35.42, 139.49) | 13.0 |

## 5. Visualizations

- `error_geographic_distribution.png`: Geographic distribution of FP/FN/TP
- `area_performance_heatmap.png`: Area-level performance

## 6. Findings

> **WARNING**: FN (1,829) exceeds TP (762). Consider lowering threshold.

> Worst areas average Recall: **15.9%**

