import json
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import matplotlib.pyplot as plt
import logging
from scipy.stats import shapiro

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CAFClusterAnalyzer:
    ZPD_REGION_BOUNDS = {
        'ZPD/2': 2.0,
        'ZPD-1': 1.0,
        'ZPD': 1.0,
        'ZPD+1': 3.0
    }

    def __init__(self, json_path):
        """Initialize the analyzer with the path to the JSON file"""
        try:
            with open(json_path, 'r') as f:
                self.data = json.load(f)

            # Validate required keys
            if 'analyzed_utterances' not in self.data:
                raise KeyError("JSON file must contain 'analyzed_utterances' key")

            self.utterances = self.data['analyzed_utterances']
            if not self.utterances:
                raise ValueError("No utterances found in the input file")

            self.features = None
            self.metadata = None
            self.linkage_matrix = None
            self.cluster_labels = None
            self.error_weights = {
                'critical': 0.4,
                'moderate': 0.2,
                'minor': 0.1
            }
            self.zones = {}  # For primary clustering only
            self.tendency_zone = None  # Store identified tendency zone

            self._version = "1.0.0"  # For state versioning
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find input file: {json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {json_path}")

    def save_state(self, path):
        """Save analysis state after primary clustering"""
        state = {
            'version': self._version,
            'features': self.features.tolist() if self.features is not None else None,
            'metadata': self.metadata,
            'linkage_matrix': self.linkage_matrix.tolist() if self.linkage_matrix is not None else None,
            'cluster_labels': self.cluster_labels.tolist() if self.cluster_labels is not None else None,
            'zones': {
                k: {'features': [f.tolist() for f in v['features']],
                    'indices': v['indices'],
                    'metadata': v['metadata']}
                for k, v in self.zones.items()
            },
            'tendency_zone': self.tendency_zone
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, json_path, state_path):
        """Load pre-computed primary clustering state"""
        analyzer = cls(json_path)  # Maintain original data reference
        with open(state_path, 'r') as f:
            state = json.load(f)
        if state.get('version', '0.0.0') != analyzer._version:
            print(f"Warning: State file version ({state.get('version', '0.0.0')}) differs from current version ({analyzer._version})")
        analyzer.features = np.array(state['features']) if state.get('features') is not None else None
        analyzer.metadata = state.get('metadata')
        analyzer.linkage_matrix = np.array(state['linkage_matrix']) if state.get('linkage_matrix') is not None else None
        analyzer.cluster_labels = np.array(state['cluster_labels']) if state.get('cluster_labels') is not None else None
        analyzer.zones = {}
        for zone, data in state['zones'].items():
            analyzer.zones[zone] = {
                'features': [np.array(feature) for feature in data['features']],
                'indices': data['indices'],
                'metadata': data['metadata']
            }
        analyzer.tendency_zone = state.get('tendency_zone')
        return analyzer

    def clear_unnecessary_data(self):
        """Clear data not needed between stages to free memory"""
        unnecessary_attrs = ['data', 'utterances']
        for attr in unnecessary_attrs:
            if hasattr(self, attr):
                delattr(self, attr)

    def extract_features(self):
        """Extract features for clustering:
        - Complexity score
        - Accuracy score
        - Normalized error counts by severity (critical, moderate, minor)
        """
        self.features = []
        self.metadata = []

        for utterance in self.utterances:
            logger.debug(f"\nProcessing utterance: {utterance.get('original', '')}")

            # Extract existing features
            complexity = utterance.get('complexity', {}).get('score', 0.0)
            accuracy = utterance.get('accuracy', {})
            accuracy_score = accuracy.get('score', 0.0)  # Direct path to accuracy score

            # Get error counts by severity
            error_counts = {
                'critical': 0,
                'moderate': 0,
                'minor': 0
            }

            # Extract error counts from each clause's errors_found
            for clause in utterance.get("clauses", []):
                clause_errors = clause.get("errors_found", [])
                for error in clause_errors:
                    severity = error.get("severity", "").lower()
                    if severity in error_counts:
                        error_counts[severity] += 1

            # Normalize error counts by total possible errors (3 types * max weight)
            max_error_score = sum(self.error_weights.values())
            normalized_errors = [
                error_counts['critical'] * self.error_weights['critical'] / max_error_score,
                error_counts['moderate'] * self.error_weights['moderate'] / max_error_score,
                error_counts['minor'] * self.error_weights['minor'] / max_error_score
            ]

            # Create feature vector (removed fluency score)
            feature_vector = [
                complexity,
                accuracy_score,
                normalized_errors[0],  # Critical errors
                normalized_errors[1],  # Moderate errors
                normalized_errors[2]   # Minor errors
            ]

            logger.debug(f"Feature vector components:")
            logger.debug(f"  complexity: {complexity} ({type(complexity)})")
            logger.debug(f"  accuracy_score: {accuracy_score} ({type(accuracy_score)})")
            logger.debug(f"  normalized_errors: {normalized_errors} ({[type(x) for x in normalized_errors]})")

            # Store metadata
            metadata = {
                'original': utterance.get('original', ''),
                'correct': utterance.get('correct', ''),
                'clauses': utterance.get('clauses', []),
                'errors': error_counts,
                'pattern_analysis': utterance.get('pattern_analysis', [])
            }

            self.features.append(feature_vector)
            self.metadata.append(metadata)
            logger.debug(f"Final error counts for utterance: {error_counts}")

        self.features = np.array(self.features)
        return self.features

    def calculate_optimal_clusters(self):
        """Determine optimal number of clusters and log silhouette scores"""
        if len(self.features) < 2:
            return 1

        max_clusters = min(10, len(self.features) - 1)
        silhouette_scores = []
        print("\nCalculating Silhouette Scores:")
        print("-----------------------------")

        for n_clusters in range(2, max_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clustering.fit_predict(self.features)
            score = silhouette_score(self.features, cluster_labels)
            silhouette_scores.append(score)
            print(f"  {n_clusters} clusters: {score:.3f}")

        # Use KneeLocator to find the optimal number of clusters
        kl = KneeLocator(
            range(2, max_clusters + 1),
            silhouette_scores,
            curve='concave',
            direction='increasing'
        )

        optimal_k = kl.elbow if kl.elbow else 2
        print(f"\nOptimal number of clusters: {optimal_k}")
        print(f"Best silhouette score: {max(silhouette_scores):.3f}\n")

        return self.validate_cluster_count(optimal_k, silhouette_scores, self.features)

    def validate_cluster_count(self, math_k, silhouette_scores, features):
        """Validate mathematical cluster count with linguistic criteria.

        Args:
            math_k: The mathematically optimal number of clusters from KneeLocator
            silhouette_scores: List of silhouette scores for different k values
            features: The feature matrix used for clustering

        Returns:
            int: Validated number of clusters that ensures reasonable cluster sizes
        """
        min_cluster_size = 3  # Minimum utterances per cluster for meaningful analysis
        max_k = len(features) // min_cluster_size

        # Ensure we don't create clusters that are too small
        validated_k = min(math_k, max_k)

        # Always have at least 2 clusters unless we have very few utterances
        if validated_k < 2 and len(features) >= 4:
            validated_k = 2

        return validated_k

    def perform_clustering(self):
        """Perform hierarchical clustering and group into zones"""
        if len(self.features) < 2:
            raise ValueError("Need at least 2 utterances to perform clustering")

        # Create linkage matrix
        self.linkage_matrix = linkage(self.features, method='ward')

        # Determine optimal number of clusters
        n_clusters = self.calculate_optimal_clusters()

        # Perform clustering
        self.clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='euclidean',
            linkage='ward'
        )
        self.cluster_labels = self.clustering.fit_predict(self.features)

        # Group into zones
        self._group_into_zones()

        # Identify tendency zone
        self.tendency_zone = self.determine_tendency_zone()

        return self.cluster_labels

    def _group_into_zones(self):
        """Organize utterances into primary zones"""
        self.zones = {}
        for i, label in enumerate(self.cluster_labels):
            zone_key = str(label)  # Just use the number as the key
            if zone_key not in self.zones:
                self.zones[zone_key] = {
                    'features': [],
                    'indices': [],
                    'metadata': []
                }
            self.zones[zone_key]['features'].append(self.features[i])
            self.zones[zone_key]['indices'].append(i)
            self.zones[zone_key]['metadata'].append(self.metadata[i])

    def generate_dendrogram(self, output_path):
        """Generate and save dendrogram"""
        plt.figure(figsize=(10, 7))
        plt.title('Primary Clustering Dendrogram')
        dendrogram(self.linkage_matrix)
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.savefig(output_path)
        plt.close()

    def _write_error_analysis(self, file, metadata):
        """Write detailed error analysis"""
        logger.debug(f"\nWriting error analysis for: {metadata.get('original', '')}")
        has_errors = False

        # Write clause-level errors
        for clause in metadata.get('clauses', []):
            if isinstance(clause, dict):
                logger.debug(f"Processing clause: {clause.get('text', '')}")
                logger.debug(f"Errors found in clause: {clause.get('errors_found', [])}")
                for error in clause.get('errors_found', []):
                    has_errors = True
                    category = error.get('category', 'Sentence Structure')
                    severity = error.get('severity', 'minor')
                    file.write(f"      - {category} ({severity})\n")
                    logger.debug(f"Wrote error: {category} ({severity})")

        if not has_errors:
            file.write("      None (error-free)\n")
            logger.debug("No errors found, wrote error-free")

    def _analyze_error_patterns(self, group):
        """Analyze error patterns in a group of utterances."""
        error_patterns = {}
        total_utterances = len(group)  # Get total utterances for ratio calculation

        for metadata in group:
            # Analyze clause-level errors
            for clause in metadata.get('clauses', []):
                if isinstance(clause, dict):
                    for error in clause.get('errors_found', []):
                        if isinstance(error, dict):
                            category = error.get('category', 'Sentence Structure')
                            error_patterns[category] = error_patterns.get(category, 0) + 1
                        else:
                            print(f"WARNING in _analyze_error_patterns: Found non-dict item in 'errors_found': {repr(error)}")

        # Convert raw counts to ratios
        if total_utterances > 0:
            for category in error_patterns:
                error_patterns[category] = error_patterns[category] / total_utterances

        return error_patterns

    def _generate_zone_name(self, avg_caf, group):
        """Generate descriptive name for a zone."""
        # Determine complexity level
        if avg_caf[0] < 0.3:
            c_level = "Simple"
        elif avg_caf[0] < 0.4:
            c_level = "Moderate"
        else:
            c_level = "Complex"

        # Determine accuracy level
        if avg_caf[1] < 0.4:
            a_level = "Inaccurate"
        elif avg_caf[1] < 0.7:
            a_level = "Moderately Accurate"
        else:
            a_level = "Accurate"

        # Get error patterns and sort by frequency ratio
        error_patterns = self._analyze_error_patterns(group)
        if error_patterns:
            # Sort error types by frequency ratio
            sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:4]
            error_summary = ", ".join([f"{err} ({score:.2f})" for err, score in sorted_errors])
            return f"{c_level} {a_level} ({error_summary})"
        else:
            return f"{c_level} {a_level}"

    def generate_cluster_analysis(self, output_path):
        """Generate analysis report in JSON format"""
        output_data = {
            "version": self._version,
            "global_stats": {},
            "tendency_zone": {},
            "clusters": []
        }

        # Calculate global statistics
        global_features = np.concatenate([np.array(zone['features']) for zone in self.zones.values()])
        global_mean = np.mean(global_features, axis=0)
        global_std = np.std(global_features, axis=0)

        # Aggregate raw error counts from metadata
        raw_errors = {'critical': 0, 'moderate': 0, 'minor': 0}
        for meta in self.metadata:
            for key in raw_errors:
                raw_errors[key] += meta['errors'].get(key, 0)
        num_utts = len(self.metadata)
        global_error_distribution = {
            "critical": {"mean": round(raw_errors['critical'] / num_utts, 2), "std": 0.0},
            "moderate": {"mean": round(raw_errors['moderate'] / num_utts, 2), "std": 0.0},
            "minor": {"mean": round(raw_errors['minor'] / num_utts, 2), "std": 0.0},
        }
        output_data["global_stats"] = {
            "total_utterances": num_utts,
            "feature_averages": {
                "complexity": {"mean": round(float(global_mean[0]), 2), "std": round(float(global_std[0]), 2)},
                "accuracy": {"mean": round(float(global_mean[1]), 2), "std": round(float(global_std[1]), 2)},
                "error_distribution": global_error_distribution
            }
        }

        # Tendency Zone Analysis
        tendency_features = np.array(self.zones[self.tendency_zone]['features'])
        tendency_avg = np.mean(tendency_features, axis=0)
        tendency_std = np.std(tendency_features, axis=0)
        tendency_distance = float(np.linalg.norm(tendency_avg - global_mean))

        output_data["tendency_zone"] = {
            "zone_id": self.tendency_zone,
            "size": len(tendency_features),
            "distance_from_global_mean": round(tendency_distance, 2),
            "feature_averages": {
                "complexity": {"mean": round(float(tendency_avg[0]), 2), "std": round(float(tendency_std[0]), 2)},
                "accuracy": {"mean": round(float(tendency_avg[1]), 2), "std": round(float(tendency_std[1]), 2)},
                "error_distribution": {
                    "critical": {"mean": round(float(tendency_avg[2]), 2), "std": round(float(tendency_std[2]), 2)},
                    "moderate": {"mean": round(float(tendency_avg[3]), 2), "std": round(float(tendency_std[3]), 2)},
                    "minor": {"mean": round(float(tendency_avg[4]), 2), "std": round(float(tendency_std[4]), 2)}
                }
            }
        }

        # Individual Zone Analyses
        for zone in sorted(self.zones.keys()):
            features = np.array(self.zones[zone]['features'])
            metadata = self.zones[zone]['metadata']
            avg_caf = np.mean(features, axis=0)
            std_caf = np.std(features, axis=0)

            print(f"\nZone {zone} Metrics:")
            print(f"  Complexity mean: {avg_caf[0]:.3f}")
            print(f"  Complexity std: {std_caf[0]:.3f}")

            # Calculate distance from global mean
            distance = np.linalg.norm(avg_caf - global_mean)

            print(f"  Size: {len(features)}")
            print(f"  Distance from mean: {distance:.3f}")
            print(f"  Composite score: {len(features) - distance:.3f}")

            # Build utterance list for this zone
            utterances = []
            for idx, utterance in enumerate(metadata):
                utterance_data = {
                    "original": utterance.get("original", ""),
                    "corrected": utterance.get("correct", ""),
                    "metrics": {
                        "C": round(float(features[idx][0]), 2),
                        "A": round(float(features[idx][1]), 2),
                        "E": {
                            "critical": round(float(features[idx][2]), 2),
                            "moderate": round(float(features[idx][3]), 2),
                            "minor": round(float(features[idx][4]), 2)
                        }
                    },
                    "errors": {
                        "critical": round(float(features[idx][2]), 2),
                        "moderate": round(float(features[idx][3]), 2),
                        "minor": round(float(features[idx][4]), 2)
                    },
                    "clauses": utterance.get("clauses", []),
                    "pattern_analysis": utterance.get("pattern_analysis", [])
                }
                utterances.append(utterance_data)

            zone_data = {
                "zone_id": zone,
                "is_tendency_zone": zone == self.tendency_zone,
                "proficiency_tier": self._generate_zone_name(avg_caf, metadata),
                "size": len(features),
                "distance_from_global_mean": round(distance, 2),
                "feature_averages": {
                    "complexity": {"mean": round(float(avg_caf[0]), 2), "std": round(float(std_caf[0]), 2)},
                    "accuracy": {"mean": round(float(avg_caf[1]), 2), "std": round(float(std_caf[1]), 2)},
                    "error_distribution": {
                        "critical": {"mean": round(float(avg_caf[2]), 2), "std": round(float(std_caf[2]), 2)},
                        "moderate": {"mean": round(float(avg_caf[3]), 2), "std": round(float(std_caf[3]), 2)},
                        "minor": {"mean": round(float(avg_caf[4]), 2), "std": round(float(std_caf[4]), 2)}
                    }
                },
                "utterances": utterances
            }
            output_data["clusters"].append(zone_data)

        # Write to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    def determine_tendency_zone(self):
        """Determine the tendency zone based on size, silhouette score, and proximity to global mean"""
        global_features = np.concatenate([np.array(zone['features']) for zone in self.zones.values()])
        global_mean = np.mean(global_features, axis=0)

        tendency_zone = None
        best_score = float('-inf')

        print("\nTendency Zone Analysis:")
        print("----------------------")

        for zone in sorted(self.zones.keys()):
            features = np.array(self.zones[zone]['features'])
            size = len(features)
            avg_caf = np.mean(features, axis=0)
            std_caf = np.std(features, axis=0)

            print(f"\nZone {zone} Metrics:")
            print(f"  Complexity mean: {avg_caf[0]:.3f}")
            print(f"  Complexity std: {std_caf[0]:.3f}")

            # Calculate distance from global mean
            distance = np.linalg.norm(avg_caf - global_mean)

            print(f"  Size: {size}")
            print(f"  Distance from mean: {distance:.3f}")
            print(f"  Composite score: {size - distance:.3f}")

            if size - distance > best_score:
                best_score = size - distance
                tendency_zone = zone
                print(f"  -> New best tendency zone!")

        print(f"\nFinal Tendency Zone: Zone {tendency_zone}")
        print(f"Composite Score: {best_score:.3f}")
        print(f"Selected Metrics:")
        print(f"  Complexity mean: {np.mean(self.zones[tendency_zone]['features'], axis=0)[0]:.3f}")
        print(f"  Complexity std: {np.std(self.zones[tendency_zone]['features'], axis=0)[0]:.3f}")

        return tendency_zone

    def define_learning_regions(self):
        """Define learning regions based on tendency zone statistics"""
        # Get tendency zone complexity values
        complexity_values = [f[0] for f in self.zones[self.tendency_zone]['features']]

        # Check for normality
        is_normal = self._check_normal_distribution(complexity_values)

        if is_normal:
            # Use mean and standard deviation for normal distributions
            mean_complexity = np.mean(complexity_values)
            std_complexity = np.std(complexity_values)

            regions = {
                'ZPD/2': (mean_complexity - 2*std_complexity, mean_complexity - std_complexity),
                'ZPD-1': (mean_complexity - std_complexity, mean_complexity),
                'ZPD': (mean_complexity, mean_complexity + std_complexity),
                'ZPD+1': (mean_complexity + std_complexity, mean_complexity + 3*std_complexity)
            }

            print("\nLearning Region Definitions (Normal Distribution):")
            print(f"  Average Complexity: {mean_complexity:.3f}")
            print(f"  Std Deviation: {std_complexity:.3f}")
        else:
            # Use median and IQR for non-normal distributions
            print("Warning: Complexity distribution non-normal, using median-based bounds")

            median_complexity = np.median(complexity_values)
            q1 = np.percentile(complexity_values, 25)
            q3 = np.percentile(complexity_values, 75)
            iqr = q3 - q1

            regions = {
                'ZPD/2': (median_complexity - 2*iqr, median_complexity - iqr),
                'ZPD-1': (median_complexity - iqr, median_complexity),
                'ZPD': (median_complexity, median_complexity + iqr),
                'ZPD+1': (median_complexity + iqr, median_complexity + 3*iqr)
            }

            print("\nLearning Region Definitions (Non-Normal Distribution):")
            print(f"  Median Complexity: {median_complexity:.3f}")
            print(f"  IQR: {iqr:.3f}")

        # Print region bounds
        print("\nRegion Bounds:")
        for region, (lower, upper) in regions.items():
            print(f"  {region}: [{lower:.3f}, {upper:.3f}]")

        return regions

    def _check_normal_distribution(self, data):
        stat, p = shapiro(data)
        return p > 0.05

    @property
    def global_avg(self):
        return np.mean(np.concatenate([zone['features'] for zone in self.zones.values()]), axis=0)

    @property
    def global_std(self):
        return np.std(np.concatenate([zone['features'] for zone in self.zones.values()]), axis=0)

    @property
    def tendency_zone_avg(self):
        return np.mean(self.zones[self.tendency_zone]['features'], axis=0)

    @property
    def tendency_zone_std(self):
        return np.std(self.zones[self.tendency_zone]['features'], axis=0)
