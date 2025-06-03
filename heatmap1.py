import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

r = np.linspace(0,140,5)
sens = [65,64,63,62,60]


x = np.linspace(-150, 150, 1500)
y = np.linspace(-150, 150, 1500)
X, Y = np.meshgrid(x, y)
f_interp = interp1d(r,sens,kind='cubic',bounds_error=False)
P = np.sqrt(X**2 + Y**2)
Z = f_interp(P)
fig = plt.figure( figsize=(9,8) )
ax1 = fig.add_subplot(1,1,1)

heat_map1 = ax1.imshow(Z,  origin='lower',cmap="jet",extent=[x.min(), x.max(), y.min(), y.max()], vmin=min(sens), # ここで最小値を指定
    vmax=max(sens))  # ここで最大値を指定)

fig.colorbar(heat_map1, ax=ax1,label='Temperature')

ax1.scatter(r, np.zeros_like(r), c=sens, cmap='jet', s=10, edgecolors='black', marker='o')
for i, val in enumerate(sens):
    # X座標: r[i], Y座標: 0 (X軸上)
    # valを文字列に変換 (formatを使って小数点以下を調整することも可能)
#     if i%5 == 0:
        ax1.annotate(f'{val:.1f}', (r[i], 0), # テキストとデータ座標
                 textcoords="offset points", # オフセットの単位
                 xytext=(0, 10),             # データ点からY方向に10ポイント上に表示
                 ha='center',                # 水平方向中央揃え
                 va='bottom',                # 垂直方向下揃え
                 fontsize=10,                 # フォントサイズ
                 color='black',              # 文字色
                 weight='bold',              # 太字
                 zorder=4)                   # 散布図マーカーより手前に表示
    
plt.xlabel('mm') # X軸ラベル
plt.ylabel('mm') # Y軸ラベル
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# --- 1. 特定の規則性を持ったXY座標と温度値のデータ生成 ---
# 中心に1点、次の半径に8点、16点、24点 (合計 1+8+16+24 = 49点)
# 各半径に割り当てる値を定義
radii = [0, 50, 100, 150] # 例: 各層の半径
points_distribution = [1, 8, 16, 24] # 各半径での点の数

x_points = []
y_points = []
data = []

# 各半径層のデータを生成
for i, r_val in enumerate(radii):
    num_points = points_distribution[i]

    if r_val == 0: # 中心点 (0,0)
        x_points.append(0.0)
        y_points.append(0.0)
        temp_val = 65 + np.random.randn() * 0.5 # 中心点の温度 (ノイズ少し)
        data.append(temp_val)
    else: # その他の半径層
        # 各半径で均等な角度を生成
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        for angle in angles:
            x_points.append(r_val * np.cos(angle))
            y_points.append(r_val * np.sin(angle))

            # 温度値を生成（例: 中心から離れるほど温度が低くなる傾向 + ノイズ）
            # データ生成の例: 半径が大きくなるほど温度が下がる
            temp_val = 65 - (r_val / 150) * 35 + np.random.randn() * 3
            data.append(temp_val)

x_points = np.array(x_points)
y_points = np.array(y_points)
data = np.array(data)

print(f"生成されたデータ点の総数: {len(x_points)}") # 49点になることを確認

# --- 2. 補間先のグリッドを生成 ---
# ヒートマップの範囲をデータの最大半径に合わせて設定
max_radius_for_grid = np.max(radii) * 1.1 # 最大半径150より少し広めに
num_grid_points = 1000 # グリッドの解像度 (適度に設定)

x_grid = np.linspace(-max_radius_for_grid, max_radius_for_grid, num_grid_points)
y_grid = np.linspace(-max_radius_for_grid, max_radius_for_grid, num_grid_points)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# --- 3. XY座標系で補間を実行 ---
# method='cubic' は滑らかな結果を生成しますが、データが疎な領域ではNaNを生成することがあります。
# NaNが出てヒートマップに白い領域ができる場合、'linear' を試すか、データ範囲を調整してください。
interpolated_Z = griddata(
    (x_points, y_points), # 補間元のデータ点のXY座標
    data,         # 補間元のデータ点の温度値
    (X_grid, Y_grid),     # 補間先のグリッドのXY座標
    method='cubic'        # 補間方法
)

# --- 4. 円形のマスクを作成 ---
# グリッドの各点から中心までの距離を計算
r_grid = np.sqrt(X_grid**2 + Y_grid**2)
# ウェハーの半径は最後のデータ層の最大半径 (例: 150) を使う
wafer_radius = np.max(radii) # データが最も広がっている半径
circular_mask = r_grid <= wafer_radius * 1.05 # ウェハーの半径より少しだけ外側まで表示する

# --- 5. マスクを適用して、円形以外の部分を NaN にする ---
interpolated_Z[~circular_mask] = np.nan

# --- 6. 結果のプロット ---
fig = plt.figure(figsize=(9, 8)) # 図のサイズを調整
ax = fig.add_subplot(1, 1, 1)

# マスクを適用したヒートマップの描画
heat_map = ax.imshow(
    interpolated_Z,
    cmap='jet', # カラーマップ
    origin='lower', # 原点を左下に
    extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
    vmin=min(data), 
    vmax=max(data)
)

fig.colorbar(heat_map, ax=ax, label='Temperature') # カラーバー

# 元のデータポイントを重ねて表示
ax.scatter(
    x_points,
    y_points,
    c=data, # 温度値で色付け
    cmap='jet', # ヒートマップと同じカラーマップ
    s=5, # マーカーサイズを小さくして、ラベルが重なりにくいように調整
    edgecolors='black', # マーカーの縁の色
    linewidth=0.5, # 縁の太さ
    zorder=3, # ヒートマップの上に表示
#     label='Original Data Points'
)

# データラベルの追加 (点数が多いため、中心付近以外は重なる可能性あり)
# 必要に応じて、一部の点のみラベルを表示するか、フォントサイズを調整してください。
# 例: 半径100以上の点のみにラベルを付けるなど
for i in range(len(x_points)):
    # r_val = np.sqrt(x_points[i]**2 + y_points[i]**2)
    # if r_val > 90: # 例: 半径90より外側の点のみラベル表示
    ax.annotate(
        f'{data[i]:.1f}', # 温度値を小数点以下1桁で表示
        (x_points[i], y_points[i]),
        textcoords='offset points', # オフセット単位
        xytext=(3, 3), # データ点からのオフセット (小さく調整)
        ha='center', # 水平方向中央揃え
        fontsize=8, # フォントサイズ (小さく調整)
        color='black', # ラベルの色
        weight='bold', # 太字
        zorder=4 # 最前面に表示
    )

# plt.title('Wafer Temperature Distribution (Circular Display with Specified Points)') # タイトル
plt.xlabel('mm') # X軸ラベル
plt.ylabel('mm') # Y軸ラベル
plt.gca().set_aspect('equal', adjustable='box') # アスペクト比を1:1に設定
# plt.legend() # 凡例を表示
plt.show()