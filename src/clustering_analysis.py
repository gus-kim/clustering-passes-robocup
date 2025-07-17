from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
import hdbscan
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull
from tabulate import tabulate

# Configurações visuais
discrete_cmap = ListedColormap(plt.cm.tab20.colors)
plt.style.use('seaborn-v0_8-darkgrid')

def load_and_preprocess_data(file_path="pass_features2.csv", features=None):
    """Carrega e pré-processa os dados."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if features is None:
        features = [
            'ball_speed', 'ball_acceleration', 'possession_change',
            'teammates_in_pass_range', 'opponents_in_pass_range',
            'best_pass_alignment', 'best_pass_distance',
            'pressure_on_ball', 'space_around_ball',
            'field_zone', 'player_speed',
            'player_body_angle', 'player_neck_angle'
        ]

    X = df[features]
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return df, X_scaled, features

def calculate_group_metrics(X_group, group_name=""):
    """Calcula métricas estatísticas para um grupo de dados."""
    if len(X_group) == 0:
        return None
        
    metrics = {
        'mean': np.mean(X_group, axis=0),
        'variance': np.var(X_group, axis=0),
        'cov_matrix': np.cov(X_group, rowvar=False),
        'size': len(X_group)
    }
    
    try:
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_group)
        if len(X_pca) > 3:
            hull = ConvexHull(X_pca)
            metrics['volume'] = hull.volume
        else:
            metrics['volume'] = 0
    except:
        metrics['volume'] = 0
    
    return metrics

def plot_initial_comparison(X_scaled, pass_mask):
    """Plota comparação inicial para todos os métodos de clusterização"""
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_
    
    # Métodos de clusterização
    methods = [
        ("Agglomerative", AgglomerativeClustering(n_clusters=10)),
        ("KMeans", KMeans(n_clusters=10, random_state=42)),
        ("HDBSCAN", hdbscan.HDBSCAN(min_cluster_size=80)),
        ("DBSCAN", DBSCAN(eps=1.0, min_samples=80))
    ]
    
    for method_name, clusterer in methods:
        # Aplica clusterização
        cluster_labels = clusterer.fit_predict(X_scaled)
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        
        # Configurações específicas para cada método
        if method_name == "Agglomerative":
            # Figura 1: Clusterização Agglomerative
            fig1 = plt.figure(figsize=(10, 8))
            ax1 = fig1.add_subplot(111, projection='3d')
            cluster_colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
            for cluster_id, color in zip(unique_clusters, cluster_colors):
                mask = cluster_labels == cluster_id
                ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                           color=color, label=f'Cluster {cluster_id}', alpha=0.6, s=20)
            ax1.set_title(f'Clusterização {method_name}\n' +
                         f'PCA1: {explained_var[0]*100:.1f}%, ' +
                         f'PCA2: {explained_var[1]*100:.1f}%, ' +
                         f'PCA3: {explained_var[2]*100:.1f}%')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            
            # Figura 2: Passes reais
            fig2 = plt.figure(figsize=(10, 8))
            ax2 = fig2.add_subplot(111, projection='3d')
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                       color='gray', alpha=0.1, s=10, label='Não-passes')
            ax2.scatter(X_pca[pass_mask, 0], X_pca[pass_mask, 1], X_pca[pass_mask, 2],
                       color='red', alpha=0.8, s=30, label='Passes')
            ax2.set_title('Passes Reais')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            
            # Figura 3: Sobreposição
            fig3 = plt.figure(figsize=(10, 8))
            ax3 = fig3.add_subplot(111, projection='3d')
            for cluster_id, color in zip(unique_clusters, cluster_colors):
                mask = cluster_labels == cluster_id
                ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                           color=color, alpha=0.4, s=20, label=f'Cluster {cluster_id}')
            ax3.scatter(X_pca[pass_mask, 0], X_pca[pass_mask, 1], X_pca[pass_mask, 2],
                       color='red', marker='x', s=50, label='Passes')
            ax3.set_title('Sobreposição: Clusters e Passes')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            
            # Figura 4: Dendrograma
            fig4 = plt.figure(figsize=(10, 8))
            ax4 = fig4.add_subplot(111)
            sample_idx = np.random.choice(len(X_scaled), size=100, replace=False)
            Z = linkage(X_scaled[sample_idx], 'ward')
            dendrogram(Z, truncate_mode='lastp', p=12, ax=ax4)
            ax4.set_title('Dendrograma (Amostra de 100 pontos)')
            ax4.set_xlabel('Índice do Ponto')
            ax4.set_ylabel('Distância')
            plt.tight_layout()
            plt.show()
            
        else:
            # Para outros métodos (KMeans, HDBSCAN, DBSCAN)
            
            # Figura 1: Clusterização
            fig1 = plt.figure(figsize=(10, 8))
            ax1 = fig1.add_subplot(111, projection='3d')
            if method_name in ["HDBSCAN", "DBSCAN"]:
                cluster_colors = ['gray' if c == -1 else plt.cm.tab20(c % 20) 
                                for c in unique_clusters]
            else:
                cluster_colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
            
            for cluster_id, color in zip(unique_clusters, cluster_colors):
                if isinstance(color, str):
                    color = [color]  # Para compatibilidade com scatter
                mask = cluster_labels == cluster_id
                label = 'Ruído' if cluster_id == -1 else f'Cluster {cluster_id}'
                ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                           color=color, label=label, alpha=0.6, s=20)
            ax1.set_title(f'Clusterização {method_name}')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            
            # Figura 2: Passes reais
            fig2 = plt.figure(figsize=(10, 8))
            ax2 = fig2.add_subplot(111, projection='3d')
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                       color='gray', alpha=0.1, s=10, label='Não-passes')
            ax2.scatter(X_pca[pass_mask, 0], X_pca[pass_mask, 1], X_pca[pass_mask, 2],
                       color='red', alpha=0.8, s=30, label='Passes')
            ax2.set_title('Passes Reais')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            
            # Figura 3: Sobreposição
            fig3 = plt.figure(figsize=(10, 8))
            ax3 = fig3.add_subplot(111, projection='3d')
            for cluster_id, color in zip(unique_clusters, cluster_colors):
                if isinstance(color, str):
                    color = [color]  # Para compatibilidade com scatter
                mask = cluster_labels == cluster_id
                label = 'Ruído' if cluster_id == -1 else f'Cluster {cluster_id}'
                ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                           color=color, alpha=0.4, s=20, label=label)
            ax3.scatter(X_pca[pass_mask, 0], X_pca[pass_mask, 1], X_pca[pass_mask, 2],
                       color='red', marker='x', s=50, label='Passes')
            ax3.set_title('Sobreposição')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

def plot_cluster_comparison(X_scaled, cluster_labels, pass_mask, method_name="Agglomerative"):
    """Função simplificada que chama a nova visualização"""
    plot_initial_comparison(X_scaled, pass_mask)

def plot_comparison(X_pca, group1_mask, group2_mask, title1="Grupo 1", title2="Grupo 2"):
    """Plota comparação entre dois grupos."""
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X_pca[group1_mask, 0], X_pca[group1_mask, 1], X_pca[group1_mask, 2], 
                color='blue', alpha=0.6, s=20, label=title1)
    ax1.scatter(X_pca[group2_mask, 0], X_pca[group2_mask, 1], X_pca[group2_mask, 2], 
                color='red', alpha=0.6, s=20, label=title2)
    ax1.set_title('Comparação 3D')
    ax1.legend()
    
    ax2 = fig.add_subplot(132)
    ax2.scatter(X_pca[group1_mask, 0], X_pca[group1_mask, 1], 
                color='blue', alpha=0.6, s=20, label=title1)
    ax2.scatter(X_pca[group2_mask, 0], X_pca[group2_mask, 1], 
                color='red', alpha=0.6, s=20, label=title2)
    ax2.set_title('Comparação 2D')
    ax2.legend()
    
    ax3 = fig.add_subplot(133)
    sns.kdeplot(X_pca[group1_mask, 0], color='blue', label=title1, ax=ax3)
    sns.kdeplot(X_pca[group2_mask, 0], color='red', label=title2, ax=ax3)
    ax3.set_title('Distribuição PCA1')
    ax3.legend()
    
    plt.suptitle(f"Comparação: {title1} vs {title2}", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_clustering_results(X_pca, original_mask, new_labels, pass_mask, method_name):
    """Plota resultados da clusterização para um método específico."""
    fig = plt.figure(figsize=(18, 6))
    
    # Get the indices where original_mask is True
    similar_indices = np.where(original_mask)[0]
    
    # --- Gráfico 1: 3D ---
    ax1 = fig.add_subplot(131, projection='3d')
    for label in np.unique(new_labels):
        if label == -1:  # Skip noise points if any
            continue
        # Get indices of points in this cluster
        cluster_indices = similar_indices[new_labels == label]
        ax1.scatter(X_pca[cluster_indices, 0], X_pca[cluster_indices, 1], X_pca[cluster_indices, 2], 
                   color=discrete_cmap(label), alpha=0.6, s=20, label=f'Cluster {label}')
    ax1.set_title(f'Clusterização {method_name.upper()}')
    ax1.legend()
    
    # --- Gráfico 2: Passes reais ---
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(X_pca[pass_mask, 0], X_pca[pass_mask, 1], X_pca[pass_mask, 2], 
               color='red', alpha=0.6, s=20, label='Passes reais')
    ax2.set_title('Passes reais')
    ax2.legend()
    
    # --- Gráfico 3: Sobreposição 2D ---
    ax3 = fig.add_subplot(133)
    for label in np.unique(new_labels):
        if label == -1:  # Skip noise points if any
            continue
        # Get indices of points in this cluster
        cluster_indices = similar_indices[new_labels == label]
        ax3.scatter(X_pca[cluster_indices, 0], X_pca[cluster_indices, 1], 
                   color=discrete_cmap(label), alpha=0.6, s=20, label=f'Cluster {label}')
    ax3.scatter(X_pca[pass_mask, 0], X_pca[pass_mask, 1], 
               color='red', alpha=0.6, s=20, label='Passes reais')
    ax3.set_title('Sobreposição 2D')
    ax3.legend()
    
    plt.suptitle(f"Clusterização {method_name.upper()} no Grupo Similar", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_final_comparison(X_pca, cluster_mask, pass_mask, best_method, similarity_percent):
    """Plota comparação final com sobreposição de clusters e passes."""
    fig = plt.figure(figsize=(18, 6))
    
    common_points = cluster_mask & pass_mask
    cluster_only = cluster_mask & ~pass_mask
    pass_only = pass_mask & ~cluster_mask
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X_pca[cluster_only, 0], X_pca[cluster_only, 1], X_pca[cluster_only, 2], 
                color='blue', alpha=0.6, s=20, label='Cluster apenas')
    ax1.scatter(X_pca[pass_only, 0], X_pca[pass_only, 1], X_pca[pass_only, 2], 
                color='red', alpha=0.6, s=20, label='Passes apenas')
    ax1.scatter(X_pca[common_points, 0], X_pca[common_points, 1], X_pca[common_points, 2], 
                color='green', alpha=0.8, s=30, label='Sobreposição')
    ax1.set_title(f'Comparação Final ({best_method})')
    ax1.legend()
    
    ax2 = fig.add_subplot(132)
    ax2.scatter(X_pca[cluster_only, 0], X_pca[cluster_only, 1], 
                color='blue', alpha=0.6, s=20, label='Cluster apenas')
    ax2.scatter(X_pca[pass_only, 0], X_pca[pass_only, 1], 
                color='red', alpha=0.6, s=20, label='Passes apenas')
    ax2.scatter(X_pca[common_points, 0], X_pca[common_points, 1], 
                color='green', alpha=0.8, s=30, label='Sobreposição')
    ax2.set_title('Visão 2D')
    ax2.legend()
    
    ax3 = fig.add_subplot(133)
    sns.kdeplot(X_pca[cluster_mask, 0], color='blue', label='Cluster', ax=ax3)
    sns.kdeplot(X_pca[pass_mask, 0], color='red', label='Passes', ax=ax3)
    ax3.set_title(f'Distribuição Comparada (Sobreposição: {similarity_percent:.2f}%)')
    ax3.legend()
    
    plt.suptitle(f"Melhor Cluster vs Passes Reais (Método: {best_method})", y=1.02)
    plt.tight_layout()
    plt.show()

def apply_clustering(X, method="hdbscan"):
    """Aplica diferentes métodos de clusterização."""
    if method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        labels = clusterer.fit_predict(X)
    elif method == "agglo":
        clusterer = AgglomerativeClustering(n_clusters=3)
        labels = clusterer.fit_predict(X)
    elif method == "kmeans":
        clusterer = KMeans(n_clusters=2, random_state=42)  # Always k=2 for kmeans
        labels = clusterer.fit_predict(X)
    elif method == "dbscan":
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        labels = clusterer.fit_predict(X)
    else:
        raise ValueError("Método desconhecido")
    
    return labels

def calculate_similarity_score(cluster_mean, pass_metrics):
    """Calcula score de similaridade entre cluster e passes."""
    mean_distance = euclidean_distances([pass_metrics['mean']], [cluster_mean])[0][0]
    return 1 / (1 + mean_distance)  # Transforma distância em score (0-1)

def final_plot(X_pca, best_subcluster_mask, pass_mask, best_method):
    """
    Plota comparação final entre o melhor cluster encontrado e os passes reais
    """
    # Configurações visuais
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(18, 8))
    
    # 1. Gráfico do cluster encontrado vs dados totais
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot todos os pontos em cinza
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
               color='gray', alpha=0.1, s=10, label='Outros dados')
    
    # Plot do melhor cluster em azul
    ax1.scatter(X_pca[best_subcluster_mask, 0], X_pca[best_subcluster_mask, 1], X_pca[best_subcluster_mask, 2],
               color='blue', alpha=0.6, s=20, label=f'Cluster {best_method}')
    
    # Plot sobreposição com passes em verde
    overlap_mask = best_subcluster_mask & pass_mask
    ax1.scatter(X_pca[overlap_mask, 0], X_pca[overlap_mask, 1], X_pca[overlap_mask, 2],
               color='green', alpha=0.8, s=30, label='Sobreposição com passes')
    
    ax1.set_title(f'Melhor Cluster ({best_method}) vs Dados Totais')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Gráfico dos passes reais vs dados totais
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot todos os pontos em cinza
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
               color='gray', alpha=0.1, s=10, label='Outros dados')
    
    # Plot passes reais em vermelho
    ax2.scatter(X_pca[pass_mask, 0], X_pca[pass_mask, 1], X_pca[pass_mask, 2],
               color='red', alpha=0.8, s=30, label='Passes reais')
    
    ax2.set_title('Passes Reais vs Dados Totais')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle("Comparação Final: Cluster Encontrado vs Passes Reais", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

def find_similar_groups(df, X_scaled, features, n_clusters=10):
    """Pipeline completo de análise."""
    pass_mask = df['is_pass'] == 1
    
    # ======================================================================
    # 0. VISUALIZAÇÃO INICIAL COMPLETA
    # ======================================================================
    print("\n[0] Visualização Inicial Completa:")
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = agglo.fit_predict(X_scaled)
    plot_cluster_comparison(X_scaled, cluster_labels, pass_mask)
    
    # ======================================================================
    # 1. ANÁLISE PRINCIPAL (MESMO FLUXO ORIGINAL)
    # ======================================================================
    # Métricas fixas dos passes
    pass_metrics = {
        'mean': np.array([
            1.70218097e+00, 3.10183455e+00, 3.93397896e+00, -4.41417737e-01,
            0.00000000e+00, -4.25592145e-01, -3.79658973e-01, -3.66292275e-01,
            -1.71673942e-01, 2.73705071e-01, 2.51425828e-16, 0.00000000e+00,
            -7.44797343e-17
        ]),
        'variance': np.array([
            2.75698954e-01, 8.45732890e-01, 3.86541844e-29, 1.97215226e-31,
            0.00000000e+00, 2.49600521e-31, 2.49600521e-31, 1.22819601e+00,
            2.73773116e-01, 6.23373851e-01, 8.75111523e-62, 0.00000000e+00,
            7.44452511e-63
        ]),
        'size': 107,
        'volume': 46.5954
    }
    
    # Encontrar cluster mais similar
    cluster_metrics = {}
    similarity_scores = {}
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_metrics[cluster_id] = calculate_group_metrics(X_scaled[cluster_mask], f"Cluster {cluster_id}")
        if cluster_metrics[cluster_id] is not None:
            similarity_scores[cluster_id] = calculate_similarity_score(
                cluster_metrics[cluster_id]['mean'], pass_metrics)
    
    # Tabela de similaridade (now as plot)
    print("\nMostrando tabela de similaridade...")
    plot_similarity_table(similarity_scores, cluster_metrics)
    
    most_similar = max(similarity_scores, key=similarity_scores.get)
    similar_mask = cluster_labels == most_similar
    X_similar = X_scaled[similar_mask]
    
    # Redução para visualização
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # 1. Comparação inicial
    print("\n[1] Comparação entre grupo similar e passes reais:")
    plot_comparison(X_pca, similar_mask, pass_mask, 
                   f"Cluster {most_similar} (similar)", "Passes reais")
    
    # ======================================================================
    # 2. APLICAÇÃO DE DIFERENTES MÉTODOS DE CLUSTERIZAÇÃO
    # ======================================================================
    methods = ["agglo", "kmeans"]
    best_metric_score = -1
    best_overlap = -1
    best_labels = None
    best_method = ""
    best_subcluster_mask = None

    for method in methods:
        print(f"\n[2] Clusterização com {method.upper()} no grupo similar:")
        # Aplica clusterização
        labels = apply_clustering(X_similar, method)
        
        # Plot resultados da clusterização (mantido original)
        plot_clustering_results(X_pca, similar_mask, labels, pass_mask, method)
        
        # Calcula métricas para cada subcluster
        subcluster_metrics = {}
        metric_scores = {}  # Similaridade com métricas de passe
        overlap_scores = {}  # Porcentagem de sobreposição com passes reais
        
        for subcluster_id in np.unique(labels):
            if subcluster_id == -1:  # Ignora outliers
                continue
                
            mask = labels == subcluster_id
            subcluster_metrics[subcluster_id] = calculate_group_metrics(X_similar[mask])
            
            # Calcula similaridade com métricas de passe
            metric_scores[subcluster_id] = calculate_similarity_score(
                subcluster_metrics[subcluster_id]['mean'], pass_metrics)
            
            # Calcula porcentagem de sobreposição com passes reais
            subcluster_mask = np.zeros(len(X_scaled), dtype=bool)
            subcluster_mask[similar_mask] = labels == subcluster_id
            intersection = np.sum(subcluster_mask & pass_mask)
            union = np.sum(subcluster_mask | pass_mask)
            overlap_scores[subcluster_id] = (intersection / union) * 100 if union > 0 else 0
        
        if not metric_scores:
            print(f"Nenhum subcluster válido encontrado com {method}")
            continue
        
        # MOSTRA SIMILARIDADE PARA TODOS OS SUBCLUSTERS
        print("\nSimilaridade com métricas de passe para cada subcluster:")
        for sub_id, score in metric_scores.items():
            print(f"Subcluster {sub_id}: {score:.4f} (Sobreposição: {overlap_scores[sub_id]:.2f}%)")
        
        # Encontra o melhor subcluster baseado na similaridade com as métricas
        best_subcluster = max(metric_scores.items(), key=lambda x: x[1])[0]
        current_metric_score = metric_scores[best_subcluster]
        current_overlap = overlap_scores[best_subcluster]
        
        # Cria máscara para o melhor subcluster deste método
        subcluster_mask = np.zeros(len(X_scaled), dtype=bool)
        subcluster_mask[similar_mask] = labels == best_subcluster
        
        print(f"\nMelhor subcluster: {best_subcluster}")
        print(f"Similaridade com métricas de passe: {current_metric_score:.4f}")
        print(f"Porcentagem de sobreposição com passes reais: {current_overlap:.2f}%")
        
        # Plotar comparação (mantido original)
        plot_comparison(X_pca, subcluster_mask, pass_mask,
                    f"Subcluster {best_subcluster} ({method})", "Passes reais")
        
        # Atualiza o melhor global baseado na similaridade com métricas
        if current_metric_score > best_metric_score:
            best_metric_score = current_metric_score
            best_overlap = current_overlap
            best_labels = labels
            best_method = method
            best_subcluster_mask = subcluster_mask

    # ======================================================================
    # 3. VISUALIZAÇÃO FINAL COM MELHOR CLUSTER (mantido original)
    # ======================================================================
    if best_subcluster_mask is not None:
        print(f"\n[3] Melhor método encontrado: {best_method}")
        print(f"Similaridade com métricas de passe: {best_metric_score:.4f}")
        print(f"Porcentagem de sobreposição com passes reais: {best_overlap:.2f}%")
        
        # Plot final
        plot_final_comparison(X_pca, best_subcluster_mask, pass_mask, best_method, best_overlap)
        final_plot(X_pca, best_subcluster_mask, pass_mask, best_method)
    else:
        print("Nenhum cluster válido encontrado em nenhum método")
        
        
def plot_similarity_table(similarity_scores, cluster_metrics):
    """Plota tabela de similaridade como figura matplotlib."""
    # Prepare table data
    table_data = []
    for cluster_id, score in similarity_scores.items():
        table_data.append([
            cluster_id,
            f"{score:.4f}",
            f"{(score * 100):.2f}%",
            cluster_metrics[cluster_id]['size']
        ])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=["Cluster", "Score", "Similaridade", "Tamanho"],
                    loc='center',
                    cellLoc='center',
                    colColours=['#f3f3f3']*4)
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Title
    plt.title("Tabela de Similaridade dos Clusters", pad=20, fontsize=14)
    plt.tight_layout()
    plt.show()

import argparse

def main():
    """Função principal para executar a análise a partir da linha de comando."""
    parser = argparse.ArgumentParser(description="Executa análise de clustering em features de passes de futebol.")
    parser.add_argument("input_file", help="Caminho para o arquivo CSV de features (da pasta CSV_filtrado).")
    args = parser.parse_args()

    print(f"Carregando dados de: {args.input_file}")

    # Carrega dados
    df, X_scaled, features = load_and_preprocess_data(args.input_file)
    
    # Verifica se há passes
    if 'is_pass' not in df.columns:
        raise ValueError("Coluna 'is_pass' não encontrada!")
    if df['is_pass'].sum() == 0:
        raise ValueError("Nenhum passe encontrado no arquivo de dados!")
    
    # Executa análise
    find_similar_groups(df, X_scaled, features)

if __name__ == "__main__":
    main()