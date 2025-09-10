# 使用範例
from analyzer import FaceClusteringAnalyzer, run_complete_analysis

# 方法1: 快速完整分析
run_complete_analysis(
    accuracy_file="accuracy_results.txt",
    confusion_matrix_file="confusion_matrix_s1ep1.xlsx",
    output_dir="face_clustering_analysis_results"
)

# 方法2: 逐步分析
analyzer = FaceClusteringAnalyzer()

# 生成所有標準視覺化
analyzer.generate_all_visualizations(
    accuracy_file="accuracy_results.txt",
    confusion_matrix_file="confusion_matrix_s1ep1.xlsx", 
    output_dir="results"
)