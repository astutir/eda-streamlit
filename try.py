import streamlit as st
import pandas as pd
import numpy as np  # Tambahkan impor ini
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Judul aplikasi
st.title('📊 Profil Data Sederhana')

# Sidebar
st.sidebar.header("Unggah File CSV")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type="csv")

if uploaded_file is not None:
    # Baca data dari file CSV
    df = pd.read_csv(uploaded_file)

    # Tampilkan informasi dasar tentang data menggunakan .info()
    st.write("## Informasi Dasar Data")
    
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Tampilkan statistik deskriptif
    st.write("## Statistik Deskriptif")
    st.write(df.describe(include='all'))

    # Analisis Missing Values
    st.write("## Analisis Missing Values")
    missing_values = df.isnull().sum()
    missing_values_percent = (missing_values / len(df)) * 100
    missing_values_df = pd.DataFrame({'Jumlah Missing': missing_values, 'Persentase': missing_values_percent})
    st.write(missing_values_df)
    
    st.write("### Visualisasi Missing Values")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title('Heatmap Missing Values', fontsize=16)
    st.pyplot(fig)

    # Pilih kolom numerik untuk visualisasi
    st.write("## Visualisasi Data Numerik")
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    selected_num_col = st.selectbox("Pilih kolom numerik", num_cols)
    hue_col_num = st.selectbox("Pilih kolom hue untuk visualisasi numerik", ['None'] + list(cat_cols))

    if selected_num_col:
        st.write(f"Distribusi untuk kolom {selected_num_col}:")

        # Histogram
        fig, ax = plt.subplots(figsize=(10, 4))
        if hue_col_num != 'None':
            long_df = pd.melt(df, id_vars=[hue_col_num], value_vars=[selected_num_col])
            sns.histplot(data=long_df, x='value', hue=hue_col_num, kde=True, ax=ax)
        else:
            sns.histplot(df[selected_num_col], kde=True, ax=ax)
        ax.set_title(f'Histogram {selected_num_col}', fontsize=14)
        st.pyplot(fig)

        # Boxplot
        fig, ax = plt.subplots(figsize=(10, 4))
        if hue_col_num != 'None':
            long_df = pd.melt(df, id_vars=[hue_col_num], value_vars=[selected_num_col])
            sns.boxplot(data=long_df, x='variable', y='value', hue=hue_col_num, ax=ax)
        else:
            sns.boxplot(y=df[selected_num_col], ax=ax)
        ax.set_title(f'Boxplot {selected_num_col}', fontsize=14)
        st.pyplot(fig)

        # Violin plot
        fig, ax = plt.subplots(figsize=(10, 4))
        if hue_col_num != 'None':
            long_df = pd.melt(df, id_vars=[hue_col_num], value_vars=[selected_num_col])
            sns.violinplot(data=long_df, x='variable', y='value', hue=hue_col_num, ax=ax)
        else:
            sns.violinplot(y=df[selected_num_col], ax=ax)
        ax.set_title(f'Violin Plot {selected_num_col}', fontsize=14)
        st.pyplot(fig)

        # Pairplot
        st.write("## Pairplot Data Numerik")
        if len(num_cols) > 1:
            pairplot_data = df[num_cols].dropna()
            g = sns.pairplot(pairplot_data)
            g.fig.suptitle('Pairplot Data Numerik', y=1.02, fontsize=16)
            st.pyplot(g.fig)
        else:
            st.write("Data tidak memiliki cukup kolom numerik untuk pairplot.")

        # Multikolinearitas
        st.write("### Indikator Multikolinearitas")
        correlation_matrix = pairplot_data.corr().abs()
        upper_triangle_matrix = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_)
        )
        high_correlation = [column for column in upper_triangle_matrix.columns if any(upper_triangle_matrix[column] > 0.8)]
        if high_correlation:
            st.write(f"Kolom-kolom yang mungkin memiliki multikolinearitas tinggi: {', '.join(high_correlation)}")
        else:
            st.write("Tidak ditemukan indikasi multikolinearitas yang tinggi di antara kolom-kolom numerik.")

    # Pilih kolom kategorikal untuk analisis
    st.write("## Visualisasi Data Kategorikal")
    selected_cat_col = st.selectbox("Pilih kolom kategorikal", cat_cols)
    hue_col_cat = st.selectbox("Pilih kolom hue untuk visualisasi kategorikal", ['None'] + list(cat_cols))

    if selected_cat_col:
        st.write(f"Frekuensi nilai untuk kolom {selected_cat_col}:")
        value_counts = df[selected_cat_col].value_counts()
        percent_counts = df[selected_cat_col].value_counts(normalize=True) * 100
        st.write(pd.DataFrame({
            'Frekuensi': value_counts,
            'Persentase': percent_counts
        }))

        st.write(f"Distribusi kategori untuk kolom {selected_cat_col}:")
        fig, ax = plt.subplots(figsize=(10, 4))
        if hue_col_cat != 'None':
            long_df = pd.melt(df, id_vars=[hue_col_cat], value_vars=[selected_cat_col])
            sns.countplot(data=long_df, y='value', hue=hue_col_cat, ax=ax, order=value_counts.index)
            for p in ax.patches:
                width = p.get_width()
                x, y = p.get_xy()
                ax.annotate(f'{width}', (x + width, y + p.get_height() / 2), va='center')
        else:
            sns.countplot(y=df[selected_cat_col], ax=ax, order=value_counts.index)
            for p in ax.patches:
                width = p.get_width()
                x, y = p.get_xy()
                ax.annotate(f'{width}', (x + width, y + p.get_height() / 2), va='center')
        ax.set_title(f'Count Plot {selected_cat_col}', fontsize=14)
        st.pyplot(fig)

    # Tampilkan hubungan antar kolom numerik
    st.write("## Hubungan Antar Kolom Numerik")
    if len(num_cols) > 1:
        st.write("Heatmap korelasi:")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt='.2f', annot_kws={"size": 10})
        ax.set_title('Heatmap Korelasi', fontsize=16)
        st.pyplot(fig)

        # Insight dari Heatmap
        st.write("### Insight dari Heatmap Korelasi")
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > 0.8:
                    col1, col2 = corr.columns[i], corr.columns[j]
                    st.write(f"Korelasi antara `{col1}` dan `{col2}` cukup tinggi (>{0.8}), mungkin terdapat multikolinearitas.")

        st.write("### Rekomendasi Penanganan")
        st.write("""
        - Jika ditemukan multikolinearitas, pertimbangkan untuk menghapus atau menggabungkan variabel atau menambah data yang memiliki korelasi tinggi.
        - Cek apakah hubungan antara kolom terkait memang relevan dengan tujuan analisis, atau jika ada variabel yang bisa diabaikan.
        """)
