# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

# %%
# ------------------------
# 0. 必要ライブラリ
# ------------------------
import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import plotly.graph_objects as go

# グローバル設定（ブランド + 列名）
glb_age = '年齢'
glb_gen = '性別'
glb_wok = '職業'
glb_cha = 'キャラ傾向'
glb_pur = '購買頻度'
glb_sty = '購入スタイル'
glb_imp = '重要視すること'
glb_sns = 'SNS利用時間'
glb_ave = '平均購入単価'

# ------------------------
# 1．Streamlit UI設定
# ------------------------
st.set_page_config(layout="wide")
st.title("💴 Van Westendorp PSM 分析アプリ")

# ------------------------
# 2．関数
# ------------------------
# 高速化
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# 価格曲線の交差点を求める
def find_intersection(y1, y2, x):
    diff = np.array(y1) - np.array(y2)
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        return None
    i = sign_change[0]
    try:
        f = interp1d(diff[i:i+2], x[i:i+2])
        return float(f(0))
    except Exception:
        return None



# %%
# ------------------------
# 3. フィルター設定
# ------------------------    

uploaded_file = st.file_uploader("📂 CSVファイルをアップロード", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.markdown("#### 🔍 絞り込みフィルター")
    col1, col2, col3 = st.columns(3)

    with col1:
        if glb_age in df.columns:
            min_age = int(df[glb_age].min())
            max_age = int(df[glb_age].max())
            selected_age_range = st.slider(f'🔍 {glb_age}', min_age, max_age, (min_age, max_age))
        else:
            selected_age_range = None

        # 性別フィルター
        if glb_gen in df.columns:
            gender_options = df[glb_gen].dropna().unique().tolist()
            if 'selected_gender' not in st.session_state:
                st.session_state.selected_gender = gender_options
            selected_gender = st.multiselect(
                f'🔍 {glb_gen}', options=gender_options, default=st.session_state.selected_gender
            )
            st.session_state.selected_gender = selected_gender
        else:
            selected_gender = None

        # 職業フィルター
        if glb_wok in df.columns:
            job_options = df[glb_wok].dropna().unique().tolist()
            if 'selected_jobs' not in st.session_state:
                st.session_state.selected_jobs = job_options
            selected_jobs = st.multiselect(
                f'🔍 {glb_wok}', options=job_options, default=st.session_state.selected_jobs
            )
            st.session_state.selected_jobs = selected_jobs
        else:
            selected_jobs = None

        if glb_ave in df.columns:
            min_price = int(df[glb_ave].min())
            max_price = int(df[glb_ave].max())
            selected_average_bands = st.slider(f'🔍 {glb_ave}', min_price, max_price, (min_price, max_price))
        else:
            selected_average_bands = None

        if glb_sns in df.columns:
            min_sns = int(df[glb_sns].min())
            max_sns = int(df[glb_sns].max())
            selected_sns = st.slider(f'🔍 {glb_sns}', min_sns, max_sns, (min_sns, max_sns))
        else:
            selected_sns = None

    with col2:
        # キャラ傾向フィルター
        if glb_cha in df.columns:
            char_options = df[glb_cha].dropna().unique().tolist()
            if 'selected_character' not in st.session_state:
                st.session_state.selected_character = char_options
            selected_character = st.multiselect(
                f'🔍 {glb_cha}', options=char_options, default=st.session_state.selected_character
            )
            st.session_state.selected_character = selected_character
        else:
            selected_character = None

        # 重要視すること フィルター
        if glb_imp in df.columns:
            imp_options = df[glb_imp].dropna().unique().tolist()
            if 'selected_importance' not in st.session_state:
                st.session_state.selected_importance = imp_options
            selected_importance = st.multiselect(
                f'🔍 {glb_imp}', options=imp_options, default=st.session_state.selected_importance
            )
            st.session_state.selected_importance = selected_importance
        else:
            selected_importance = None

        # 購買頻度フィルター
        if glb_pur in df.columns:
            freq_options = df[glb_pur].dropna().unique().tolist()
            if 'selected_frequency' not in st.session_state:
                st.session_state.selected_frequency = freq_options
            selected_frequency = st.multiselect(
                f'🔍 {glb_pur}', options=freq_options, default=st.session_state.selected_frequency
            )
            st.session_state.selected_frequency = selected_frequency
        else:
            selected_frequency = None

    with col3:

        if glb_sty in df.columns:
            style_options = df[glb_sty].dropna().unique().tolist()
            st.markdown(f'🔍 {glb_sty}')
        
            # セッションステート初期化
            for s in style_options:
                key_name = f"selected_style_{s}"
                if key_name not in st.session_state:
                    st.session_state[key_name] = True  # 初期は全選択状態
        
            colA, colB = st.columns(2)
            with colA:
                if st.button("✅ 全て選択"):
                    for s in style_options:
                        st.session_state[f"selected_style_{s}"] = True
            with colB:
                if st.button("❌ 全て解除"):
                    for s in style_options:
                        st.session_state[f"selected_style_{s}"] = False
        
            # チェックボックス
            selected_style = []
            for s in style_options:
                key_name = f"selected_style_{s}"
                checked = st.checkbox(s, key=key_name)  # value= は不要
                if checked:
                    selected_style.append(s)
        else:
            selected_style = None

    show_lines = st.checkbox("📊 指標の補助線とラベルを表示/非表示", value=True)

    # フィルター処理（順序は意味を持たず、独立して適用されます）
    filtered_df = df.copy()
    if selected_age_range:
        filtered_df = filtered_df[filtered_df[glb_age].between(*selected_age_range)]
    if selected_gender:
        filtered_df = filtered_df[filtered_df[glb_gen].isin(selected_gender)]
    if selected_jobs:
        filtered_df = filtered_df[filtered_df[glb_wok].isin(selected_jobs)]
    if selected_character:
        filtered_df = filtered_df[filtered_df[glb_cha].isin(selected_character)]
    if selected_frequency:
        filtered_df = filtered_df[filtered_df[glb_pur].isin(selected_frequency)]
    if selected_style:
        filtered_df = filtered_df[filtered_df[glb_sty].isin(selected_style)]
    if selected_importance:
        filtered_df = filtered_df[filtered_df[glb_imp].isin(selected_importance)]
    if selected_sns:
        filtered_df = filtered_df[filtered_df[glb_sns].between(*selected_sns)]
    if selected_average_bands:
        filtered_df = filtered_df[filtered_df[glb_ave].between(*selected_average_bands)]

    labels = ['too_cheap', 'cheap', 'expensive', 'too_expensive']
    brands = sorted(set(col.split('_')[0] for col in df.columns if any(lbl in col for lbl in labels)))
    tabs = st.tabs(brands)


    # %%
    # ------------------------   
    #  4．PSM分析とグラフ表示
    # ------------------------   
    results = []
    num_people = filtered_df.shape[0]
    for tab, brand in zip(tabs, brands):
        with tab:
            brand_cols = [f"{brand}_{lbl}" for lbl in labels if f"{brand}_{lbl}" in filtered_df.columns]
            df_brand = filtered_df[filtered_df[brand_cols].notnull().any(axis=1)]
            if df_brand.empty:
                st.warning(f"{brand} のデータがありません。")
                continue
            # データ準備 
            price_data = {
                label: df_brand[f"{brand}_{label}"].dropna().astype(int).values
                for label in labels if f"{brand}_{label}" in df_brand.columns
            }

            valid_arrays = [arr for arr in price_data.values() if len(arr) > 0]
            if not valid_arrays:
                st.warning("有効な価格データがありません。")
                continue

            all_prices = np.arange(
                min(np.concatenate(valid_arrays)),
                max(np.concatenate(valid_arrays)) + 1000,
                100
            )
            n = len(df_brand)

            # 累積比率計算
            cumulative = {
                'too_cheap': [np.sum(price_data.get('too_cheap', []) >= p) / n for p in all_prices],
                'cheap': [np.sum(price_data.get('cheap', []) >= p) / n for p in all_prices],
                'expensive': [np.sum(price_data.get('expensive', []) <= p) / n for p in all_prices],
                'too_expensive': [np.sum(price_data.get('too_expensive', []) <= p) / n for p in all_prices],
            }

            # 交点計算
            opp = find_intersection(cumulative['cheap'], cumulative['expensive'], all_prices)
            idp = find_intersection(cumulative['too_cheap'], cumulative['too_expensive'], all_prices)
            pme = find_intersection(cumulative['cheap'], cumulative['too_expensive'], all_prices)
            pmc = find_intersection(cumulative['expensive'], cumulative['too_cheap'], all_prices)

            # グラフ作成
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['too_cheap'])*100, name='Too Cheap', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['cheap'])*100, name='Cheap', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['expensive'])*100, name='Expensive', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=all_prices, y=np.array(cumulative['too_expensive'])*100, name='Too Expensive', line=dict(color='red')))

            if show_lines:
                for val, name, color in zip(
                    [opp, idp, pme, pmc],
                    ['OPP（最適）', 'IDP（無関心）', 'PME（上限）', 'PMC（下限）'],
                    ['purple', 'black', 'magenta', 'cyan']
                ):
                    if val:
                        fig.add_vline(x=val, line_dash='dash', line_color=color)
                        fig.add_annotation(x=val, y=50, text=name, showarrow=False, textangle=90,
                                           font=dict(color=color, size=12), bgcolor='white')

            fig.update_layout(
                title=f"{brand} - PSM分析",
                xaxis_title="価格（円）",
                yaxis_title="累積比率（%）",
                height=400,
                hovermode="x unified",
                xaxis=dict(tickformat='d')
            )

            # 結果表示
            col_info, col_graph = st.columns([1, 2])

            with col_info:
                st.markdown("#### 👇 指標")
                st.markdown(f"**{brand} の該当人数：{df_brand.shape[0]}人**")
                st.write(f"📌 **最適価格（OPP）**: {round(opp) if opp else '計算不可'} 円")
                st.write(f"📌 **無関心価格（IDP）**: {round(idp) if idp else '計算不可'} 円")
                st.write(f"📌 **価格受容範囲下限（PMC）**: {round(pmc) if pmc else '計算不可'} 円")
                st.write(f"📌 **価格受容範囲上限（PME）**: {round(pme) if pme else '計算不可'} 円")

                # データ集計
                results.append({
                    "ブランド": brand,
                    "OPP": opp,
                    "IDP": idp,
                    "PMC": pmc,
                    "PME": pme
                })

                summary_df = pd.DataFrame(results)
                brand_row = summary_df[summary_df["ブランド"] == brand]
                st.dataframe(brand_row.style.format({col: "{:.0f}" for col in brand_row.columns if col != "ブランド"}))

            with col_graph:
                st.plotly_chart(fig, use_container_width=True)

    # %%
    # ------------------------ 
    # フィルター前の計算結果（別集計）
    # ------------------------ 
    results_before_filter = []
    for brand in brands:
        brand_cols = [f"{brand}_{lbl}" for lbl in labels if f"{brand}_{lbl}" in df.columns]
        df_brand = df[df[brand_cols].notnull().any(axis=1)]
        if df_brand.empty:
            continue

        price_data = {
            label: df_brand[f"{brand}_{label}"].dropna().astype(int).values
            for label in labels if f"{brand}_{label}" in df_brand.columns
        }

        valid_arrays = [arr for arr in price_data.values() if len(arr) > 0]
        if not valid_arrays:
            continue

        all_prices = np.arange(
            min(np.concatenate(valid_arrays)),
            max(np.concatenate(valid_arrays)) + 1000,
            100
        )
        n = len(df_brand)

        cumulative = {
            'too_cheap': [np.sum(price_data.get('too_cheap', []) >= p) / n for p in all_prices],
            'cheap': [np.sum(price_data.get('cheap', []) >= p) / n for p in all_prices],
            'expensive': [np.sum(price_data.get('expensive', []) <= p) / n for p in all_prices],
            'too_expensive': [np.sum(price_data.get('too_expensive', []) <= p) / n for p in all_prices],
        }

        opp = find_intersection(cumulative['cheap'], cumulative['expensive'], all_prices)
        idp = find_intersection(cumulative['too_cheap'], cumulative['too_expensive'], all_prices)
        pme = find_intersection(cumulative['cheap'], cumulative['too_expensive'], all_prices)
        pmc = find_intersection(cumulative['expensive'], cumulative['too_cheap'], all_prices)

        results_before_filter.append({
            "ブランド": brand,
            "OPP": opp,
            "IDP": idp,
            "PMC": pmc,
            "PME": pme
        })

    st.markdown("---")
    st.markdown("#### 👇フィルター前 ブランド別 PSM 指標一覧")
    summary_df_before = pd.DataFrame(results_before_filter)
    st.markdown(f"**調査人数：{len(df)}人**")
    st.dataframe(summary_df_before.style.format({col: "{:.0f}" for col in summary_df_before.columns if col != "ブランド"}))


else:
    st.info("CSVファイルをアップロードしてください。")

# %%
