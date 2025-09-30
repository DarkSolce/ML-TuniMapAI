"""
ML-TuniMapAi - Application Streamlit Interactive
Système de prédiction et optimisation de routes en Tunisie
Version complète avec matrices de temps de trajet
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Imports conditionnels
try:
    import folium
    from streamlit_folium import folium_static
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    def geodesic(coord1, coord2):
        class Result:
            def __init__(self, km):
                self.kilometers = km
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        return Result(((lat2-lat1)**2 + (lon2-lon1)**2)**0.5 * 111)

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Configuration
st.set_page_config(
    page_title="ML-TuniMapAi",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =======================
# PARAMÈTRES DE TRANSPORT
# =======================

TRANSPORT_PARAMS = {
    'walking': {
        'base_speed_kmh': 4,
        'detour_factor': 1.0,
        'waiting_time_min': 0,
        'comfort_factor': 1.0,
        'max_reasonable_distance': 5
    },
    'bike': {
        'base_speed_kmh': 15,
        'detour_factor': 1.1,
        'waiting_time_min': 2,
        'comfort_factor': 0.8,
        'max_reasonable_distance': 20
    },
    'bus': {
        'base_speed_kmh': 25,
        'detour_factor': 1.6,
        'waiting_time_min': 12,
        'comfort_factor': 0.7,
        'max_reasonable_distance': 100
    },
    'metro': {
        'base_speed_kmh': 35,
        'detour_factor': 1.4,
        'waiting_time_min': 8,
        'comfort_factor': 0.9,
        'max_reasonable_distance': 50
    },
    'train': {
        'base_speed_kmh': 60,
        'detour_factor': 1.5,
        'waiting_time_min': 15,
        'comfort_factor': 0.8,
        'max_reasonable_distance': 300
    },
    'taxi': {
        'base_speed_kmh': 28,
        'detour_factor': 1.2,
        'waiting_time_min': 8,
        'comfort_factor': 0.9,
        'max_reasonable_distance': 100
    },
    'car': {
        'base_speed_kmh': 35,
        'detour_factor': 1.2,
        'waiting_time_min': 3,
        'comfort_factor': 1.0,
        'max_reasonable_distance': 500
    }
}

CAR_PERFORMANCE = {
    60: {'speed_factor': 0.85, 'fuel_efficiency': 1.2, 'label': 'Economy 60HP'},
    90: {'speed_factor': 1.0, 'fuel_efficiency': 1.0, 'label': 'Standard 90HP'},
    120: {'speed_factor': 1.1, 'fuel_efficiency': 0.9, 'label': 'Mid 120HP'},
    150: {'speed_factor': 1.2, 'fuel_efficiency': 0.8, 'label': 'Premium 150HP'},
    200: {'speed_factor': 1.4, 'fuel_efficiency': 0.6, 'label': 'Sport 200HP'},
    300: {'speed_factor': 1.5, 'fuel_efficiency': 0.5, 'label': 'Luxury 300HP'}
}

COST_PER_KM = {
    'walking': 0,
    'bike': 0.02,
    'bus': 0.15,
    'metro': 0.20,
    'train': 0.25,
    'taxi': 1.50,
    'car': 0.35
}

# ===================
# CLASSES DE DONNÉES
# ===================

class MLModelManager:
    """Gestionnaire des modèles ML pour la prédiction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.results = {}
        self.is_trained = False
    
    @st.cache_resource
    def train_models(_self, df, distance_matrix, max_samples=50000):
        """Entraîne les modèles ML pour la prédiction de temps"""
        if not SKLEARN_AVAILABLE:
            st.warning("scikit-learn non disponible. Fonctionnalités ML limitées.")
            return False
        
        st.info("🔄 Entraînement des modèles ML en cours...")
        progress_bar = st.progress(0)
        
        try:
            # Préparation des données
            _self._prepare_ml_dataset(df, distance_matrix)
            
            # Modes à entraîner
            modes_to_train = ['car_90hp', 'bus', 'taxi']
            
            for idx, mode in enumerate(modes_to_train):
                progress_bar.progress((idx + 1) / len(modes_to_train))
                
                # Simulation des temps pour entraînement
                y_train, y_test = _self._generate_training_data(df, distance_matrix, mode, max_samples)
                
                if len(y_train) > 100:
                    # Entraînement
                    model, scaler, metrics = _self._train_single_model(
                        _self.X_train, _self.X_test, y_train, y_test
                    )
                    
                    if model is not None:
                        _self.models[mode] = model
                        _self.scalers[mode] = scaler
                        _self.results[mode] = metrics
            
            _self.is_trained = len(_self.models) > 0
            progress_bar.empty()
            
            if _self.is_trained:
                st.success(f"✅ {len(_self.models)} modèles entraînés avec succès!")
            
            return _self.is_trained
            
        except Exception as e:
            st.error(f"Erreur lors de l'entraînement: {e}")
            progress_bar.empty()
            return False
    
    def _prepare_ml_dataset(self, df, distance_matrix):
        """Prépare le dataset pour le ML"""
        n_samples = min(len(df), 1000)  # Limitation pour performance
        indices = np.random.choice(len(df), n_samples, replace=False)
        
        features_list = []
        
        for i in indices:
            for j in indices:
                if i == j:
                    continue
                
                origin = df.iloc[i]
                dest = df.iloc[j]
                
                features = {
                    'origin_lat': origin['Latitude'],
                    'origin_lon': origin['Longitude'],
                    'origin_accessibility': origin['accessibility_score'],
                    'origin_tourism': origin['tourism_score'],
                    'dest_lat': dest['Latitude'],
                    'dest_lon': dest['Longitude'],
                    'dest_accessibility': dest['accessibility_score'],
                    'dest_tourism': dest['tourism_score'],
                    'distance_km': distance_matrix[i, j],
                    'same_governorate': int(origin['Governorate'] == dest['Governorate']),
                    'tourism_diff': abs(origin['tourism_score'] - dest['tourism_score']),
                    'accessibility_diff': abs(origin['accessibility_score'] - dest['accessibility_score'])
                }
                
                features_list.append(features)
        
        df_ml = pd.DataFrame(features_list)
        
        # Split train/test
        train_idx = int(len(df_ml) * 0.8)
        self.X_train = df_ml.iloc[:train_idx]
        self.X_test = df_ml.iloc[train_idx:]
        self.feature_columns = list(df_ml.columns)
    
    def _generate_training_data(self, df, distance_matrix, mode, max_samples):
        """Génère les données d'entraînement simulées"""
        n_train = len(self.X_train)
        n_test = len(self.X_test)
        
        # Simulation basée sur les paramètres de transport
        params = TRANSPORT_PARAMS.get('car' if 'car' in mode else mode.replace('_90hp', ''), TRANSPORT_PARAMS['car'])
        
        y_train = []
        for _, row in self.X_train.iterrows():
            distance = row['distance_km']
            base_time = (distance * params['detour_factor']) / params['base_speed_kmh']
            wait_time = params['waiting_time_min'] / 60.0
            noise = np.random.normal(0, 0.1)
            y_train.append(max(0.01, base_time + wait_time + noise))
        
        y_test = []
        for _, row in self.X_test.iterrows():
            distance = row['distance_km']
            base_time = (distance * params['detour_factor']) / params['base_speed_kmh']
            wait_time = params['waiting_time_min'] / 60.0
            noise = np.random.normal(0, 0.1)
            y_test.append(max(0.01, base_time + wait_time + noise))
        
        return np.array(y_train), np.array(y_test)
    
    def _train_single_model(self, X_train, X_test, y_train, y_test):
        """Entraîne un modèle pour un mode de transport"""
        try:
            # Normalisation
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Entraînement avec SGD (rapide et efficace)
            model = SGDRegressor(
                max_iter=1000,
                tol=1e-3,
                random_state=42,
                learning_rate='invscaling',
                eta0=0.01
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Évaluation
            y_pred = model.predict(X_test_scaled)
            
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'MAE': mae,
                'R2': r2,
                'samples': len(y_train)
            }
            
            return model, scaler, metrics
            
        except Exception as e:
            st.warning(f"Erreur entraînement modèle: {e}")
            return None, None, None
    
    def predict(self, origin_features, dest_features, mode='car_90hp'):
        """Prédit le temps de trajet"""
        if mode not in self.models:
            return None
        
        try:
            # Préparer les features
            features = {**origin_features, **dest_features}
            input_df = pd.DataFrame([features])
            
            # Assurer que toutes les colonnes sont présentes
            for col in self.feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[self.feature_columns]
            
            # Normaliser et prédire
            X_scaled = self.scalers[mode].transform(input_df)
            prediction = self.models[mode].predict(X_scaled)[0]
            
            return max(0.01, prediction)
            
        except Exception as e:
            st.warning(f"Erreur prédiction: {e}")
            return None
    
    def get_model_performance(self):
        """Retourne les performances des modèles"""
        return self.results


class TunisiaDataLoader:
    """Chargement des données"""
    
    @staticmethod
    @st.cache_data
    def load_municipalities_and_delegations(json_file_path):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            all_delegations = []
            municipalities = [data] if isinstance(data, dict) else data
            
            for muni in municipalities:
                muni_name = muni.get('Name', 'Unknown')
                muni_name_ar = muni.get('NameAr', '')
                muni_val = muni.get('Value', '')
                delegations = muni.get('Delegations', [])
                
                if not delegations:
                    all_delegations.append({
                        'Governorate': muni_name,
                        'GovernorateAr': muni_name_ar,
                        'GovernorateValue': muni_val
                    })
                else:
                    for delg in delegations:
                        row = delg.copy()
                        row['Governorate'] = muni_name
                        row['GovernorateAr'] = muni_name_ar
                        row['GovernorateValue'] = muni_val
                        all_delegations.append(row)
            
            return pd.DataFrame(all_delegations)
            
        except FileNotFoundError:
            return TunisiaDataLoader._generate_simulated_data()
    
    @staticmethod
    def _generate_simulated_data():
        governorates_info = [
            {"name": "TUNIS", "name_ar": "تونس", "center": (36.8065, 10.1815)},
            {"name": "ARIANA", "name_ar": "أريانة", "center": (36.8665, 10.1947)},
            {"name": "BEN AROUS", "name_ar": "بن عروس", "center": (36.7544, 10.2318)},
            {"name": "NABEUL", "name_ar": "نابل", "center": (36.4561, 10.7376)},
            {"name": "BIZERTE", "name_ar": "بنزرت", "center": (37.2746, 9.8740)},
            {"name": "SFAX", "name_ar": "صفاقس", "center": (34.7406, 10.7603)},
        ]
        
        np.random.seed(42)
        all_delegations = []
        
        for i, gov in enumerate(governorates_info):
            center_lat, center_lon = gov["center"]
            num_delegations = np.random.randint(30, 60)
            
            for j in range(num_delegations):
                lat_offset = np.random.normal(0, 0.08)
                lon_offset = np.random.normal(0, 0.08)
                
                delegation = {
                    'Name': f"{gov['name']} - Cite {j+1}",
                    'NameAr': f"{gov['name_ar']} حي {j+1}",
                    'Value': gov['name'],
                    'PostalCode': f"{2000 + i*100 + j}",
                    'Latitude': center_lat + lat_offset,
                    'Longitude': center_lon + lon_offset,
                    'Governorate': gov['name'],
                    'GovernorateAr': gov['name_ar'],
                    'GovernorateValue': gov['name']
                }
                all_delegations.append(delegation)
        
        return pd.DataFrame(all_delegations)


class DataPreprocessor:
    """Prétraitement des données"""
    
    @staticmethod
    def classify_zone_type(name, name_ar=""):
        name_lower = str(name).lower()
        if any(word in name_lower for word in ['cite', 'حي', 'residence']):
            return 'residential'
        elif any(word in name_lower for word in ['centre', 'commercial', 'مركز']):
            return 'commercial'
        elif any(word in name_lower for word in ['aeroport', 'مطار', 'station']):
            return 'transport'
        elif any(word in name_lower for word in ['mosque', 'جامع']):
            return 'religious'
        elif any(word in name_lower for word in ['tourist', 'beach', 'شاطئ']):
            return 'tourism'
        else:
            return 'mixed'
    
    @staticmethod
    @st.cache_data
    def preprocess_data(df):
        df = df.copy()
        
        # Nettoyage
        df['Name'] = df['Name'].str.strip()
        df['Governorate'] = df['Governorate'].str.strip()
        
        # Filtrage coordonnées
        TUNISIA_BOUNDS = {'lat_min': 30.0, 'lat_max': 38.0, 'lon_min': 7.0, 'lon_max': 12.0}
        valid_coords = (
            (df['Latitude'] >= TUNISIA_BOUNDS['lat_min']) &
            (df['Latitude'] <= TUNISIA_BOUNDS['lat_max']) &
            (df['Longitude'] >= TUNISIA_BOUNDS['lon_min']) &
            (df['Longitude'] <= TUNISIA_BOUNDS['lon_max'])
        )
        df = df[valid_coords].copy()
        
        # IDs
        df['zone_id'] = range(1, len(df) + 1)
        df['unique_id'] = df['Governorate'].str.upper() + '_' + df['zone_id'].astype(str)
        
        # Distance au centre
        tunisia_center = (df['Latitude'].mean(), df['Longitude'].mean())
        df['distance_to_national_center'] = df.apply(
            lambda row: geodesic((row['Latitude'], row['Longitude']), tunisia_center).kilometers,
            axis=1
        )
        
        # Classification zones
        df['zone_type'] = df.apply(
            lambda row: DataPreprocessor.classify_zone_type(row['Name'], row.get('NameAr', '')),
            axis=1
        )
        
        # Simulation transport
        np.random.seed(42)
        transport_probs = {
            'residential': 0.7, 'commercial': 0.85, 'transport': 0.95,
            'religious': 0.5, 'tourism': 0.6, 'mixed': 0.6
        }
        
        for transport in ['bus', 'metro', 'train', 'taxi']:
            probs = df['zone_type'].map(transport_probs)
            df[f'has_{transport}_station'] = np.random.binomial(1, probs)
        
        # Score accessibilité
        transport_score = (
            df['has_bus_station'] * 1 +
            df['has_metro_station'] * 2 +
            df['has_train_station'] * 3 +
            df['has_taxi_station'] * 0.5
        )
        
        max_distance = df['distance_to_national_center'].max()
        distance_score = 1 - (df['distance_to_national_center'] / max_distance)
        
        zone_importance = {
            'transport': 5, 'commercial': 4, 'tourism': 3,
            'residential': 1, 'religious': 1, 'mixed': 2
        }
        importance_score = df['zone_type'].map(zone_importance)
        
        total_score = transport_score * 0.4 + distance_score * 3 + importance_score * 0.6
        df['accessibility_score'] = (total_score - total_score.min()) / (total_score.max() - total_score.min()) * 10
        
        # Features touristiques
        coastal_regions = ['TUNIS', 'ARIANA', 'BEN AROUS', 'NABEUL', 'BIZERTE', 'SOUSSE', 'SFAX']
        df['landscape_beauty_score'] = df.apply(
            lambda row: 5 + (3 if row['Governorate'] in coastal_regions else 0) + 
                       (4 if row['zone_type'] == 'tourism' else 0) + np.random.normal(0, 1),
            axis=1
        ).clip(1, 10)
        
        df['cultural_interest_score'] = df.apply(
            lambda row: 3 + (4 if row['Governorate'] in ['TUNIS', 'KAIROUAN'] else 0) + 
                       np.random.normal(0, 0.8),
            axis=1
        ).clip(1, 10)
        
        df['tourism_score'] = (
            df['landscape_beauty_score'] * 0.4 +
            df['cultural_interest_score'] * 0.3 +
            df['accessibility_score'] * 0.3
        )
        
        # Clustering
        if SKLEARN_AVAILABLE:
            features_for_clustering = ['Latitude', 'Longitude', 'distance_to_national_center', 'accessibility_score']
            scaler = StandardScaler()
            clustering_data = scaler.fit_transform(df[features_for_clustering])
            kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
            df['cluster_id'] = kmeans.fit_predict(clustering_data)
        else:
            df['cluster_id'] = 0
        
        return df


class TravelTimeCalculator:
    """Calcul des temps de trajet avec matrices"""
    
    @staticmethod
    @st.cache_data
    def calculate_distance_matrix(df):
        """Calcule la matrice des distances"""
        n_points = len(df)
        distance_matrix = np.zeros((n_points, n_points))
        coords = df[['Latitude', 'Longitude']].values
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = geodesic(coords[i], coords[j]).kilometers
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    @staticmethod
    def calculate_travel_time(origin_idx, dest_idx, df, distance_matrix, transport_mode, car_hp=90):
        """Calcule le temps de trajet entre deux points"""
        if origin_idx == dest_idx:
            return {
                'time_hours': 0,
                'distance_km': 0,
                'cost_dt': 0,
                'comfort_score': 10,
                'error': None
            }
        
        distance = distance_matrix[origin_idx, dest_idx]
        params = TRANSPORT_PARAMS.get(transport_mode, TRANSPORT_PARAMS['car'])
        
        # Vérification distance maximale
        if distance > params['max_reasonable_distance']:
            return {'error': f'Distance trop grande pour {transport_mode}'}
        
        # Vérification disponibilité transport public
        if transport_mode in ['bus', 'metro', 'train']:
            origin_available = df.iloc[origin_idx].get(f'has_{transport_mode}_station', False)
            dest_available = df.iloc[dest_idx].get(f'has_{transport_mode}_station', False)
            if not (origin_available and dest_available):
                return {'error': f'{transport_mode.capitalize()} non disponible sur ce trajet'}
        
        # Calcul vitesse effective
        effective_speed = params['base_speed_kmh']
        if transport_mode == 'car':
            car_perf = CAR_PERFORMANCE.get(car_hp, CAR_PERFORMANCE[90])
            effective_speed *= car_perf['speed_factor']
        
        # Calcul temps
        travel_time = (distance * params['detour_factor']) / effective_speed
        waiting_time = params['waiting_time_min'] / 60.0
        total_time = travel_time + waiting_time
        
        # Calcul coût
        cost_per_km = COST_PER_KM.get(transport_mode, 0.35)
        cost = distance * cost_per_km
        if transport_mode == 'car':
            car_perf = CAR_PERFORMANCE.get(car_hp, CAR_PERFORMANCE[90])
            cost *= (2.0 - car_perf['fuel_efficiency'])
        
        # Score confort
        comfort = params['comfort_factor'] * 10
        if transport_mode == 'car':
            comfort += (car_hp - 90) * 0.01
        comfort = np.clip(comfort, 1, 10)
        
        return {
            'time_hours': total_time,
            'distance_km': distance,
            'cost_dt': cost,
            'comfort_score': comfort,
            'error': None
        }
    
    @staticmethod
    def find_optimal_routes(origin_idx, dest_idx, df, distance_matrix, preferences):
        """Trouve les routes optimales selon les préférences"""
        results = []
        
        # Test tous les modes
        for mode in ['walking', 'bike', 'bus', 'metro', 'train', 'taxi', 'car']:
            if mode == 'car':
                # Test plusieurs puissances
                for hp in [60, 90, 120, 150, 200, 300]:
                    result = TravelTimeCalculator.calculate_travel_time(
                        origin_idx, dest_idx, df, distance_matrix, mode, hp
                    )
                    if not result.get('error'):
                        score = (
                            preferences.get('time_weight', 0.4) * (1 / (result['time_hours'] + 0.1)) +
                            preferences.get('cost_weight', 0.3) * (1 / (result['cost_dt'] + 0.1)) +
                            preferences.get('comfort_weight', 0.3) * (result['comfort_score'] / 10)
                        )
                        results.append({
                            'mode': f"car_{hp}hp",
                            'label': CAR_PERFORMANCE[hp]['label'],
                            **result,
                            'composite_score': score
                        })
            else:
                result = TravelTimeCalculator.calculate_travel_time(
                    origin_idx, dest_idx, df, distance_matrix, mode
                )
                if not result.get('error'):
                    score = (
                        preferences.get('time_weight', 0.4) * (1 / (result['time_hours'] + 0.1)) +
                        preferences.get('cost_weight', 0.3) * (1 / (result['cost_dt'] + 0.1)) +
                        preferences.get('comfort_weight', 0.3) * (result['comfort_score'] / 10)
                    )
                    results.append({
                        'mode': mode,
                        'label': mode.capitalize(),
                        **result,
                        'composite_score': score
                    })
        
        # Tri par score
        results = sorted(results, key=lambda x: x['composite_score'], reverse=True)
        return results
    
    @staticmethod
    def find_scenic_route(origin_idx, dest_idx, df):
        """Trouve des points d'intérêt touristiques"""
        origin = df.iloc[origin_idx]
        dest = df.iloc[dest_idx]
        
        # Points touristiques
        tourist_points = df[df['tourism_score'] > df['tourism_score'].quantile(0.7)]
        
        origin_coords = (origin['Latitude'], origin['Longitude'])
        dest_coords = (dest['Latitude'], dest['Longitude'])
        
        waypoints = []
        for _, poi in tourist_points.iterrows():
            poi_coords = (poi['Latitude'], poi['Longitude'])
            
            dist_from_origin = geodesic(origin_coords, poi_coords).kilometers
            dist_from_dest = geodesic(poi_coords, dest_coords).kilometers
            direct_dist = geodesic(origin_coords, dest_coords).kilometers
            
            if dist_from_origin + dist_from_dest < direct_dist * 1.5:
                waypoints.append({
                    'idx': poi.name,
                    'name': poi['Name'],
                    'tourism_score': poi['tourism_score'],
                    'landscape_score': poi['landscape_beauty_score'],
                    'cultural_score': poi['cultural_interest_score'],
                    'latitude': poi['Latitude'],
                    'longitude': poi['Longitude']
                })
        
        waypoints = sorted(waypoints, key=lambda x: x['tourism_score'], reverse=True)[:5]
        return waypoints


# ===================
# CHARGEMENT DONNÉES
# ===================

@st.cache_resource
def load_data():
    json_path = 'data/raw/state-municipality-areas.json'
    df = TunisiaDataLoader.load_municipalities_and_delegations(json_path)
    df = DataPreprocessor.preprocess_data(df)
    distance_matrix = TravelTimeCalculator.calculate_distance_matrix(df)
    
    # Initialiser le gestionnaire ML
    ml_manager = MLModelManager()
    
    return df, distance_matrix, ml_manager


# ===================
# PAGES APPLICATION
# ===================

def show_home_page(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Zones", len(df))
    with col2:
        st.metric("Gouvernorats", df['Governorate'].nunique())
    with col3:
        st.metric("Types de Zones", df['zone_type'].nunique())
    with col4:
        st.metric("Score Accès Moyen", f"{df['accessibility_score'].mean():.1f}/10")
    
    st.markdown("### 📊 Répartition par Gouvernorat")
    gov_counts = df['Governorate'].value_counts().head(10)
    fig = px.bar(x=gov_counts.index, y=gov_counts.values,
                 title="Top 10 Gouvernorats", labels={'x': 'Gouvernorat', 'y': 'Zones'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Informations système
    st.markdown("### 🤖 Capacités du Système")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Algorithmes de Calcul**
        - Matrices de distances géodésiques
        - Simulation réseau routier
        - Optimisation multi-critères
        """)
    
    with col2:
        st.markdown("""
        **Modes de Transport**
        - 8 modes différents
        - 6 catégories de voitures
        - Paramètres personnalisables
        """)
    
    with col3:
        st.markdown("""
        **Intelligence Artificielle**
        - Modèles ML entraînables
        - Prédictions temps réel
        - Recommandations intelligentes
        """)


def show_ml_prediction_page(df, distance_matrix, ml_manager):
    st.markdown('<h2 class="sub-header">🤖 Prédiction ML Avancée</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Utilisez nos modèles de Machine Learning entraînés pour obtenir des prédictions 
    de temps de trajet plus précises basées sur l'analyse de milliers de trajets.
    </div>
    """, unsafe_allow_html=True)
    
    # Vérifier si les modèles sont entraînés
    if not ml_manager.is_trained:
        st.warning("⚠️ Modèles ML non entraînés")
        
        if st.button("🚀 Entraîner les Modèles ML", type="primary"):
            with st.spinner("Entraînement en cours... Cela peut prendre quelques minutes."):
                success = ml_manager.train_models(df, distance_matrix)
                
                if success:
                    st.success("✅ Modèles entraînés avec succès!")
                    st.rerun()
                else:
                    st.error("❌ Échec de l'entraînement")
        
        return
    
    # Afficher les performances des modèles
    st.markdown("### 📊 Performance des Modèles")
    
    perf = ml_manager.get_model_performance()
    
    if perf:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_mae = np.mean([v['MAE'] for v in perf.values()])
            st.metric("MAE Moyenne", f"{avg_mae:.2f}h", "Erreur absolue")
        
        with col2:
            avg_r2 = np.mean([v['R2'] for v in perf.values()])
            st.metric("R² Moyen", f"{avg_r2:.3f}", "Qualité du modèle")
        
        with col3:
            total_samples = sum([v['samples'] for v in perf.values()])
            st.metric("Échantillons", f"{total_samples:,}", "Données d'entraînement")
        
        # Tableau détaillé
        st.markdown("#### Détails par Mode")
        
        perf_data = []
        for mode, metrics in perf.items():
            perf_data.append({
                'Mode': mode.replace('_', ' ').title(),
                'MAE (heures)': f"{metrics['MAE']:.3f}",
                'R²': f"{metrics['R2']:.3f}",
                'Échantillons': f"{metrics['samples']:,}"
            })
        
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
    
    # Interface de prédiction
    st.markdown("### 🔮 Faire une Prédiction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Origine")
        origin_gov = st.selectbox("Gouvernorat", df['Governorate'].unique(), key="ml_orig")
        origin_opts = df[df['Governorate'] == origin_gov]
        origin_zone = st.selectbox("Zone", range(len(origin_opts)),
                                  format_func=lambda x: origin_opts.iloc[x]['Name'], key="ml_oz")
        origin_idx = origin_opts.iloc[origin_zone].name
        origin_row = df.iloc[origin_idx]
    
    with col2:
        st.subheader("🎯 Destination")
        dest_gov = st.selectbox("Gouvernorat", df['Governorate'].unique(), key="ml_dest")
        dest_opts = df[df['Governorate'] == dest_gov]
        dest_zone = st.selectbox("Zone", range(len(dest_opts)),
                                format_func=lambda x: dest_opts.iloc[x]['Name'], key="ml_dz")
        dest_idx = dest_opts.iloc[dest_zone].name
        dest_row = df.iloc[dest_idx]
    
    # Mode de transport
    mode = st.selectbox("Mode de transport", 
                       list(ml_manager.models.keys()),
                       format_func=lambda x: x.replace('_', ' ').title())
    
    if st.button("🔮 Prédire avec ML", type="primary"):
        # Préparer les features
        origin_features = {
            'origin_lat': origin_row['Latitude'],
            'origin_lon': origin_row['Longitude'],
            'origin_accessibility': origin_row['accessibility_score'],
            'origin_tourism': origin_row['tourism_score']
        }
        
        dest_features = {
            'dest_lat': dest_row['Latitude'],
            'dest_lon': dest_row['Longitude'],
            'dest_accessibility': dest_row['accessibility_score'],
            'dest_tourism': dest_row['tourism_score'],
            'distance_km': distance_matrix[origin_idx, dest_idx],
            'same_governorate': int(origin_row['Governorate'] == dest_row['Governorate']),
            'tourism_diff': abs(origin_row['tourism_score'] - dest_row['tourism_score']),
            'accessibility_diff': abs(origin_row['accessibility_score'] - dest_row['accessibility_score'])
        }
        
        # Prédiction ML
        ml_prediction = ml_manager.predict(origin_features, dest_features, mode)
        
        # Calcul standard pour comparaison
        transport_mode = 'car' if 'car' in mode else mode
        hp = 90 if 'car' in mode else None
        standard_result = TravelTimeCalculator.calculate_travel_time(
            origin_idx, dest_idx, df, distance_matrix, transport_mode, hp if hp else 90
        )
        
        if ml_prediction and not standard_result.get('error'):
            st.success("✅ Prédiction réussie!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🤖 Prédiction ML", f"{ml_prediction:.2f}h")
            
            with col2:
                st.metric("📐 Calcul Standard", f"{standard_result['time_hours']:.2f}h")
            
            with col3:
                diff = abs(ml_prediction - standard_result['time_hours'])
                st.metric("📊 Différence", f"{diff:.2f}h")
            
            # Graphique de comparaison
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Prédiction ML', 'Calcul Standard'],
                y=[ml_prediction, standard_result['time_hours']],
                marker_color=['#1f77b4', '#ff7f0e']
            ))
            
            fig.update_layout(
                title="Comparaison ML vs Standard",
                yaxis_title="Temps (heures)",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Détails
            with st.expander("📋 Détails de la prédiction"):
                st.write(f"**Distance:** {dest_features['distance_km']:.1f} km")
                st.write(f"**Même gouvernorat:** {'Oui' if dest_features['same_governorate'] else 'Non'}")
                st.write(f"**Diff. tourisme:** {dest_features['tourism_diff']:.1f}")
                st.write(f"**Diff. accessibilité:** {dest_features['accessibility_diff']:.1f}")
                
                if mode in ml_manager.results:
                    metrics = ml_manager.results[mode]
                    st.write(f"**Précision du modèle (R²):** {metrics['R2']:.3f}")
                    st.write(f"**Erreur moyenne (MAE):** {metrics['MAE']:.3f}h")


def show_prediction_page(df, distance_matrix):
    st.markdown('<h2 class="sub-header">🚗 Prédiction de Trajet</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Départ")
        origin_gov = st.selectbox("Gouvernorat", df['Governorate'].unique(), key="orig")
        origin_opts = df[df['Governorate'] == origin_gov]
        origin_zone = st.selectbox("Zone", range(len(origin_opts)),
                                  format_func=lambda x: origin_opts.iloc[x]['Name'], key="oz")
        origin_idx = origin_opts.iloc[origin_zone].name
    
    with col2:
        st.subheader("🎯 Destination")
        dest_gov = st.selectbox("Gouvernorat", df['Governorate'].unique(), key="dest")
        dest_opts = df[df['Governorate'] == dest_gov]
        dest_zone = st.selectbox("Zone", range(len(dest_opts)),
                                format_func=lambda x: dest_opts.iloc[x]['Name'], key="dz")
        dest_idx = dest_opts.iloc[dest_zone].name
    
    # Configuration préférences
    st.markdown("### ⚙️ Préférences de Voyage")
    col1, col2, col3 = st.columns(3)
    with col1:
        time_weight = st.slider("Importance Temps", 0.0, 1.0, 0.4, 0.1)
    with col2:
        cost_weight = st.slider("Importance Coût", 0.0, 1.0, 0.3, 0.1)
    with col3:
        comfort_weight = st.slider("Importance Confort", 0.0, 1.0, 0.3, 0.1)
    
    preferences = {
        'time_weight': time_weight,
        'cost_weight': cost_weight,
        'comfort_weight': comfort_weight
    }
    
    if st.button("🔮 Trouver Meilleures Routes", type="primary"):
        results = TravelTimeCalculator.find_optimal_routes(
            origin_idx, dest_idx, df, distance_matrix, preferences
        )
        
        if results:
            st.success(f"✅ {len(results)} options trouvées!")
            
            # Top 3
            st.markdown("### 🏆 Top 3 Options")
            for i, result in enumerate(results[:3], 1):
                with st.expander(f"#{i} - {result['label']} (Score: {result['composite_score']:.2f})"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("⏱️ Temps", f"{result['time_hours']:.1f}h")
                    with col2:
                        st.metric("📏 Distance", f"{result['distance_km']:.1f} km")
                    with col3:
                        st.metric("💰 Coût", f"{result['cost_dt']:.2f} DT")
                    with col4:
                        st.metric("😊 Confort", f"{result['comfort_score']:.1f}/10")
            
            # Graphique comparatif
            st.markdown("### 📊 Comparaison Détaillée")
            df_results = pd.DataFrame(results[:8])
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Temps', 'Coût', 'Confort')
            )
            
            fig.add_trace(
                go.Bar(x=df_results['label'], y=df_results['time_hours'], name='Temps'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=df_results['label'], y=df_results['cost_dt'], name='Coût'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=df_results['label'], y=df_results['comfort_score'], name='Confort'),
                row=1, col=3
            )
            
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def show_tourism_page(df, distance_matrix):
    st.markdown('<h2 class="sub-header">🎯 Routes Touristiques</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Départ")
        origin_gov = st.selectbox("Gouvernorat", df['Governorate'].unique(), key="tour_orig")
        origin_opts = df[df['Governorate'] == origin_gov]
        origin_zone = st.selectbox("Zone", range(len(origin_opts)),
                                  format_func=lambda x: origin_opts.iloc[x]['Name'], key="toz")
        origin_idx = origin_opts.iloc[origin_zone].name
    
    with col2:
        st.subheader("🎯 Destination")
        dest_gov = st.selectbox("Gouvernorat", df['Governorate'].unique(), key="tour_dest")
        dest_opts = df[df['Governorate'] == dest_gov]
        dest_zone = st.selectbox("Zone", range(len(dest_opts)),
                                format_func=lambda x: dest_opts.iloc[x]['Name'], key="tdz")
        dest_idx = dest_opts.iloc[dest_zone].name
    
    if st.button("🗺️ Générer Route Touristique", type="primary"):
        waypoints = TravelTimeCalculator.find_scenic_route(origin_idx, dest_idx, df)
        
        if waypoints:
            st.success(f"✅ {len(waypoints)} points d'intérêt trouvés!")
            
            for i, wp in enumerate(waypoints, 1):
                with st.expander(f"📍 {i}. {wp['name']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Score Tourisme", f"{wp['tourism_score']:.1f}/10")
                    with col2:
                        st.metric("Score Paysage", f"{wp['landscape_score']:.1f}/10")
                    with col3:
                        st.metric("Score Culturel", f"{wp['cultural_score']:.1f}/10")
            
            # Carte si disponible
            if FOLIUM_AVAILABLE:
                st.markdown("### 🗺️ Carte de la Route")
                origin_point = df.iloc[origin_idx]
                dest_point = df.iloc[dest_idx]
                
                all_lats = [origin_point['Latitude'], dest_point['Latitude']] + [wp['latitude'] for wp in waypoints]
                all_lons = [origin_point['Longitude'], dest_point['Longitude']] + [wp['longitude'] for wp in waypoints]
                
                m = folium.Map(location=[np.mean(all_lats), np.mean(all_lons)], zoom_start=8)
                
                # Départ
                folium.Marker(
                    [origin_point['Latitude'], origin_point['Longitude']],
                    popup=f"Départ: {origin_point['Name']}",
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(m)
                
                # Destination
                folium.Marker(
                    [dest_point['Latitude'], dest_point['Longitude']],
                    popup=f"Arrivée: {dest_point['Name']}",
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(m)
                
                # Points d'intérêt
                for i, wp in enumerate(waypoints, 1):
                    folium.Marker(
                        [wp['latitude'], wp['longitude']],
                        popup=f"POI {i}: {wp['name']}<br>Score: {wp['tourism_score']:.1f}/10",
                        icon=folium.Icon(color='blue', icon='camera')
                    ).add_to(m)
                
                # Ligne de route
                route_coords = [[origin_point['Latitude'], origin_point['Longitude']]]
                for wp in sorted(waypoints, key=lambda x: 
                    geodesic((origin_point['Latitude'], origin_point['Longitude']), 
                            (x['latitude'], x['longitude'])).kilometers):
                    route_coords.append([wp['latitude'], wp['longitude']])
                route_coords.append([dest_point['Latitude'], dest_point['Longitude']])
                
                folium.PolyLine(route_coords, color='red', weight=3, opacity=0.8).add_to(m)
                folium_static(m, width=1200, height=500)
        else:
            st.warning("Aucun point d'intérêt trouvé sur ce trajet.")


def show_map_page(df):
    st.markdown('<h2 class="sub-header">🗺️ Carte Interactive</h2>', unsafe_allow_html=True)
    
    if not FOLIUM_AVAILABLE:
        selected_govs = st.multiselect("Gouvernorats", df['Governorate'].unique(),
                                      default=df['Governorate'].unique()[:5])
        if selected_govs:
            filtered = df[df['Governorate'].isin(selected_govs)]
            fig = px.scatter(filtered, x='Longitude', y='Latitude', color='Governorate',
                           hover_name='Name', title="Carte des Zones")
            st.plotly_chart(fig, use_container_width=True)
        return
    
    selected_govs = st.multiselect("Gouvernorats", df['Governorate'].unique(),
                                  default=df['Governorate'].unique()[:5])
    
    if selected_govs:
        filtered = df[df['Governorate'].isin(selected_govs)]
        m = folium.Map(location=[filtered['Latitude'].mean(), filtered['Longitude'].mean()], zoom_start=8)
        
        colors = px.colors.qualitative.Set3
        color_map = {gov: colors[i % len(colors)] for i, gov in enumerate(selected_govs)}
        
        marker_cluster = plugins.MarkerCluster().add_to(m)
        
        for _, row in filtered.iterrows():
            popup_text = f"<b>{row['Name']}</b><br>Gouvernorat: {row['Governorate']}"
            if 'zone_type' in row:
                popup_text += f"<br>Type: {row['zone_type']}"
            if 'accessibility_score' in row:
                popup_text += f"<br>Accessibilité: {row['accessibility_score']:.1f}/10"
            
            folium.CircleMarker(
                [row['Latitude'], row['Longitude']],
                radius=4,
                popup=popup_text,
                color=color_map[row['Governorate']],
                fill=True,
                fillOpacity=0.6
            ).add_to(marker_cluster)
        
        folium_static(m, width=1200, height=600)


def show_stats_page(df):
    st.markdown('<h2 class="sub-header">📊 Statistiques Détaillées</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gov_counts = df['Governorate'].value_counts()
        fig = px.pie(values=gov_counts.values, names=gov_counts.index,
                    title="Répartition par Gouvernorat")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        zone_counts = df['zone_type'].value_counts()
        fig = px.bar(x=zone_counts.index, y=zone_counts.values,
                    title="Types de Zones")
        st.plotly_chart(fig, use_container_width=True)
    
    # Top zones touristiques
    st.markdown("### 🏆 Top 10 Zones Touristiques")
    top_zones = df.nlargest(10, 'tourism_score')[['Name', 'Governorate', 'tourism_score', 'accessibility_score']]
    st.dataframe(top_zones.reset_index(drop=True), use_container_width=True)
    
    # Distribution des scores
    st.markdown("### 📈 Distribution des Scores")
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'Score Touristique', 'Score Accessibilité', 
        'Score Paysage', 'Score Culturel'
    ))
    
    fig.add_trace(go.Histogram(x=df['tourism_score'], nbinsx=20, name='Tourisme'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df['accessibility_score'], nbinsx=20, name='Accessibilité'), row=1, col=2)
    fig.add_trace(go.Histogram(x=df['landscape_beauty_score'], nbinsx=20, name='Paysage'), row=2, col=1)
    fig.add_trace(go.Histogram(x=df['cultural_interest_score'], nbinsx=20, name='Culturel'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def show_analysis_page(df, distance_matrix):
    st.markdown('<h2 class="sub-header">🔬 Analyse Avancée</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Zones Totales", len(df))
    with col2:
        st.metric("Distance Moyenne", f"{distance_matrix[distance_matrix > 0].mean():.1f} km")
    with col3:
        st.metric("Distance Max", f"{distance_matrix.max():.1f} km")
    with col4:
        st.metric("Modes Transport", len(TRANSPORT_PARAMS))
    
    # Matrice de corrélation
    st.markdown("### 🔗 Corrélations entre Scores")
    score_cols = ['tourism_score', 'accessibility_score', 'landscape_beauty_score', 'cultural_interest_score']
    corr_matrix = df[score_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                    text_auto=True,
                    color_continuous_scale='RdBu',
                    title="Matrice de Corrélation")
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse par type de zone
    st.markdown("### 🏗️ Analyse par Type de Zone")
    zone_analysis = df.groupby('zone_type').agg({
        'tourism_score': 'mean',
        'accessibility_score': 'mean',
        'landscape_beauty_score': 'mean',
        'cultural_interest_score': 'mean'
    }).round(2)
    
    fig = px.bar(zone_analysis, barmode='group',
                 title="Scores Moyens par Type de Zone")
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution géographique
    st.markdown("### 🌍 Distribution Géographique")
    fig = px.scatter(df, x='Longitude', y='Latitude', 
                     color='cluster_id',
                     size='tourism_score',
                     hover_name='Name',
                     title="Clustering Géographique")
    st.plotly_chart(fig, use_container_width=True)


def show_transport_comparison_page(df, distance_matrix):
    st.markdown('<h2 class="sub-header">🚦 Comparaison des Transports</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Analysez et comparez les différents modes de transport disponibles en Tunisie
    avec leurs caractéristiques détaillées.
    </div>
    """, unsafe_allow_html=True)
    
    # Tableau comparatif
    st.markdown("### 📋 Tableau Comparatif des Modes de Transport")
    
    transport_data = []
    for mode, params in TRANSPORT_PARAMS.items():
        transport_data.append({
            'Mode': mode.capitalize(),
            'Vitesse (km/h)': params['base_speed_kmh'],
            'Facteur Détour': params['detour_factor'],
            'Attente (min)': params['waiting_time_min'],
            'Confort': params['comfort_factor'],
            'Distance Max (km)': params['max_reasonable_distance']
        })
    
    df_transport = pd.DataFrame(transport_data)
    st.dataframe(df_transport, use_container_width=True, hide_index=True)
    
    # Comparaison visuelle
    st.markdown("### 📊 Comparaison Visuelle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(df_transport, x='Mode', y='Vitesse (km/h)',
                     title="Vitesses Moyennes")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(df_transport, x='Mode', y='Confort',
                     title="Scores de Confort")
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance des voitures
    st.markdown("### 🚗 Performance des Voitures par Puissance")
    
    car_data = []
    for hp, perf in CAR_PERFORMANCE.items():
        car_data.append({
            'Puissance (HP)': hp,
            'Label': perf['label'],
            'Facteur Vitesse': perf['speed_factor'],
            'Efficacité Carburant': perf['fuel_efficiency']
        })
    
    df_cars = pd.DataFrame(car_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(df_cars, x='Puissance (HP)', y='Facteur Vitesse',
                      markers=True, title="Facteur de Vitesse vs Puissance")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(df_cars, x='Puissance (HP)', y='Efficacité Carburant',
                      markers=True, title="Efficacité Carburant vs Puissance")
        st.plotly_chart(fig, use_container_width=True)


# ===================
# MAIN APPLICATION
# ===================

def main():
    st.markdown('<h1 class="main-header">🗺️ ML-TuniMapAi</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Système de prédiction et optimisation de routes en Tunisie</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choisissez une page", 
        ["🏠 Accueil", 
         "🚗 Prédiction de Trajet",
         "🤖 Prédiction ML Avancée",
         "🎯 Routes Touristiques",
         "🗺️ Carte Interactive", 
         "📊 Statistiques",
         "🔬 Analyse Avancée",
         "🚦 Comparaison Transports"])
    
    # Chargement des données
    try:
        with st.spinner("Chargement des données..."):
            df, distance_matrix, ml_manager = load_data()
        
        st.sidebar.success(f"✅ {len(df)} zones chargées")
        st.sidebar.info(f"📍 {df['Governorate'].nunique()} gouvernorats")
        
        # Indicateur ML
        if ml_manager.is_trained:
            st.sidebar.success(f"🤖 {len(ml_manager.models)} modèles ML actifs")
        else:
            st.sidebar.warning("⚠️ Modèles ML non entraînés")
        
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {e}")
        st.stop()
    
    # Navigation
    if page == "🏠 Accueil":
        show_home_page(df)
    elif page == "🚗 Prédiction de Trajet":
        show_prediction_page(df, distance_matrix)
    elif page == "🤖 Prédiction ML Avancée":
        show_ml_prediction_page(df, distance_matrix, ml_manager)
    elif page == "🎯 Routes Touristiques":
        show_tourism_page(df, distance_matrix)
    elif page == "🗺️ Carte Interactive":
        show_map_page(df)
    elif page == "📊 Statistiques":
        show_stats_page(df)
    elif page == "🔬 Analyse Avancée":
        show_analysis_page(df, distance_matrix)
    elif page == "🚦 Comparaison Transports":
        show_transport_comparison_page(df, distance_matrix)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        if st.checkbox("Afficher détails"):
            import traceback
            st.code(traceback.format_exc())