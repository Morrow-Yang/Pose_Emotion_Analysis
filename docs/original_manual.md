Pose–Emotion 指标计算说明（可复现版）
一、基础约定
关键点来源：AlphaPose / COCO 17 keypoints
每个关键点包含 (x, y, confidence)，统一置信度阈值：conf ≥ 0.2
 关键点编号（COCO 17）
名称	说明
nose	鼻
left_eye / right_eye	眼
left_ear / right_ear	耳
left_shoulder / right_shoulder	肩
left_elbow / right_elbow	肘
left_wrist / right_wrist	腕
left_hip / right_hip	髋
left_knee / right_knee	膝
left_ankle / right_ankle	踝

躯干中心（torso_center）：左右肩与左右髋的几何平均点
torso_center = mean(
    left_shoulder,
    right_shoulder,
    left_hip,
    right_hip
)

躯干尺度（torso_scale）：左右肩中心与左右髋中心的垂直距离，用于归一化；消除拍摄距离 / 人体大小差异
torso_scale = vertical distance between
              mean(shoulders) and mean(hips)

二、Shoulder Width（肩宽）
关键点：left_shoulder, right_shoulder
计算：|x_left_shoulder − x_right_shoulder| / torso_scale
含义：横向身体展开程度，数值越大越“占空间”
三、Contraction Index（身体收缩/展开指数）
关键点：left_wrist, right_wrist, left_ankle, right_ankle
计算：四肢末端到躯干中心距离的平均值，再除以 torso_scale
含义：末端距离越大，身体越展开；越小则越收缩
四、Arm Span / Arm Extension（上肢展开度）
Arm Span：|x_left_wrist − x_right_wrist| / torso_scale
Arm Extension：左右 wrist–shoulder 距离的平均值 / torso_scale
含义：衡量手臂张开或贴近身体的程度
五、Elbow Angle（肘部角度）
关键点：shoulder, elbow, wrist
计算：肩–肘–腕构成的夹角
含义：反映手臂弯曲或抬起程度
六、Head–Body Vertical Offset（headby）
关键点：nose（或头部中心）, torso_center
计算：(y_head − y_torso_center) / torso_scale
含义：头部相对躯干的垂直位置变化，作为辅助信号
七、valid_kp（关键点可见数量）
关键点：全部 17 个
计算：confidence ≥ 0.2 的关键点数量
含义：姿态观测质量指标，仅用于筛选与控制
Python代码(其实只要能得到上面那些 COCO 17 keypoints就可以直接接着我的得到答案了)
metrics.py：批处理提取指标 + 缺失率统计 + CSV 输出
 1) metrics.py：怎么用（批处理 + 缺失率 + CSV）
输入数据格式（必须满足）
数据目录结构应为：
DATA_ROOT/
  angry/
    alphapose-results.json
  sad/
    alphapose-results.json
  surprise/
    alphapose-results.json
  neutral/
    alphapose-results.json
  ...

最常用命令
python metrics.py \
  --root DATA_ROOT \
  --json_name alphapose-results.json \
  --outdir out_metrics
out_metrics/
  per_sample_metrics.csv
  summary_missingness.csv
  summary_counts.csv
重点看这两个：
	summary_counts.csv
¡	平均 valid_kp 是否 ≥ 10
	summary_missingness.csv
¡	contraction / arm_span / shoulder_width 缺失率是否太高
如果缺失率很高，说明：
	图像裁剪太狠
	或 AlphaPose 检测质量差

run_v1_v5_analysis.py：读取 per-sample 指标 CSV，按 v1–v5 不同严格度跑 ANOVA/η²，并输出跨版本一致性表
2) run_v1_v5_analysis.py：怎么用（接 v1–v5）
这个脚本会做：
定义一个 “v1→v5 逐渐更严格” 的筛选梯度（默认用 valid_kp_min 收紧）
每个版本对每个特征做：
单因素 ANOVA
η²（effect size）
输出：
v1_results.csv … v5_results.csv
cross_variant_consistency.csv（跨版本方向一致性）
最常用命令
python run_v1_v5_analysis.py \
  --metrics_csv out_metrics/per_sample_metrics.csv \
  --outdir out_analysis \
  --label_col label
out_analysis/
  v1_results.csv
  v2_results.csv
  ...
  cross_variant_consistency.csv
其中：
	v*_results.csv
¡	每个特征的 p-value + η²
	cross_variant_consistency.csv
¡	看一个结论在 v1–v5 中方向是否一致（很重要）

 

附录：常见坑与错误用法
1. contraction 指标方向搞反
contraction/limb-to-torso distance 越大通常表示四肢末端离躯干中心越远（更展开）。若你把它解释成“越大越收缩”，结论会完全反向。建议在实现后用几张样例可视化验证。
2. y 轴方向不一致导致 headby 符号相反
不同库/坐标系可能 y 轴向下为正（图像坐标）或向上为正（数学坐标）。headby=(y_head - y_torso)/scale 的正负会随坐标系翻转。请在文档/代码中明确 y 轴定义，并统一到一个约定。
3. 忘记做尺度归一化（torso_scale）
所有距离/跨度类指标必须除以 torso_scale（或等价人体尺度），否则会把拍摄距离、分辨率差异误当成情绪差异。
4. 低置信度关键点直接参与计算
必须先用 conf_thr=0.2 判定可见性；对于不可见点应当：跳过该点或返回 NaN，并记录 valid_kp。不要用 (0,0) 或上一次位置填补，否则会引入严重偏差。
5. torso_center / torso_scale 计算不稳
肩或髋缺失时，torso_center/scale 会漂移。建议：若 core 关键点不足（例如 <3/4 可见），直接判为无效样本。
6. angle 计算数值不稳定（除零/浮点误差）
角度计算需对向量长度做 epsilon 保护；对 acos 输入做 clip 到 [-1,1]，避免 NaN。
7. arm_span 对裁剪敏感
arm_span 依赖双腕同时可见；如果数据集经常半身裁剪，应优先使用 arm_extension（wrist–shoulder）并报告缺失率。
8. 混用均值与中位数导致“版本不一致”
如果你在一个数据集用 mean，在另一个用 median，效应可能改变。建议在跨数据集复现时固定汇总策略。
9. 坐标未去除人物整体平移
大多数指标用相对距离可抵消平移；但若你直接用绝对坐标（例如 head_y），会引入构图偏差。推荐使用相对 torso_center 的位移。
10. 忽视缺失率与样本选择偏差
计算每个指标时记录有效样本数与缺失率；不同情绪若缺失率差异很大，需作为 bias 讨论或协变量控制。


新增：CAER 全视频（10 fps）处理与主人人对齐流程

1) 抽帧（推荐 10 fps，保持情绪子目录）：
  ffmpeg -i raw/CAER/<emotion>/<video>.mp4 -vf fps=10 outputs/frames/CAER/<emotion>/%04d.png

2) AlphaPose：对抽帧目录跑检测，生成 alphapose-results.json（与 CAER-S 相同层级）。

3) 主人人筛选（几何/时序用同一主体）：
  python scripts/features/filter_top1_alphapose.py \
    --input  outputs/alphapose/outputs/CAER/<emotion>/alphapose-results.json \
    --output outputs/alphapose/outputs/CAER/<emotion>/alphapose-results.top1.json
  几何、时序分析均使用 *.top1.json，保证 image_id 对齐同一人。

4) 几何特征（v4）：
  python scripts/analysis/analysis_v4.py \
    --root outputs/alphapose/outputs/CAER \
    --json_name alphapose-results.top1.json \
    --out_dir outputs/analysis/analysis/caer_v4

5) 时序特征（含 image_id，便于合并）：
  python scripts/innovation/temporal_motion_analysis.py \
    --root outputs/alphapose/outputs/CAER \
    --out  outputs/analysis/temporal/caer_v1

6) 融合几何+时序：在 (emotion, image_id) 上 merge，得到 pose_features_caer_with_temporal.csv。

7) 融合分析：
  python scripts/analysis/analysis_v4.py \
    --root outputs/alphapose/outputs/CAER \
    --precomputed_csv outputs/analysis/analysis/caer_v4/pose_features_caer_with_temporal.csv \
    --out_dir outputs/analysis/analysis/caer_v4_with_temporal \
    --json_name alphapose-results.top1.json

提示：若 KMeans 出现 MKL 内存泄露警告，运行前可设 OMP_NUM_THREADS=6。


