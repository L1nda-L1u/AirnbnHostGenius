# =============================================
# 快速生成基线模型对比图（R²对比）
# =============================================
# 
# 如果已经运行过 evaluate_all_baseline_models.R，
# 可以直接加载结果并生成图表
#
# =============================================

library(ggplot2)
library(dplyr)
library(tidyr)

# 如果存在结果文件，直接加载
if (file.exists("baseline_models_summary.csv")) {
  cat("Loading existing results...\n")
  summary_df <- read.csv("baseline_models_summary.csv", stringsAsFactors = FALSE)
} else {
  cat("Results file not found. Please run evaluate_all_baseline_models.R first.\n")
  stop("No results to plot.")
}

# 生成R²对比图（更美观的版本）
p_r2 <- ggplot(summary_df, aes(x = reorder(Model, R2), y = R2, fill = R2)) +
  geom_bar(stat = "identity", width = 0.7) +
  coord_flip() +
  scale_fill_gradient2(low = "lightblue", mid = "steelblue", high = "darkblue", 
                       midpoint = median(summary_df$R2)) +
  labs(
    title = "Baseline Models Comparison: R² Score",
    subtitle = "Higher R² indicates better model performance",
    x = "Model",
    y = "R² Score",
    caption = "Five baseline models evaluated on test set"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 12, color = "gray40"),
    plot.caption = element_text(hjust = 0.5, size = 10, color = "gray60"),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 12, face = "bold"),
    legend.position = "none",
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  geom_text(aes(label = sprintf("%.4f", R2)), 
            hjust = -0.1, size = 4, fontface = "bold") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))

# 保存R²对比图
ggsave("baseline_models_r2_comparison.png", p_r2, 
       width = 10, height = 7, dpi = 300, bg = "white")
cat("Saved: baseline_models_r2_comparison.png\n")

# 打印结果
cat("\nModel Performance Summary:\n")
cat("==========================\n")
print(summary_df[order(-summary_df$R2), ])

# 显示图表
print(p_r2)

