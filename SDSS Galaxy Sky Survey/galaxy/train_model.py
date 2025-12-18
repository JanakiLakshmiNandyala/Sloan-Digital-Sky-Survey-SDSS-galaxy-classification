
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import warnings

warnings.filterwarnings('ignore')

def train_model():
    # 1. Load Data
    print("Loading data...")
    try:
        df = pd.read_csv('galaxy_data.csv')
    except FileNotFoundError:
        print("Error: galaxy_data.csv not found.")
        return

    # 2. Data Preparation
    print("Pre-processing data...")
    # Handling missing values (though synthetic data has none, good practice)
    df.dropna(inplace=True)

    # Handling Categorical Data: subclass
    le = LabelEncoder()
    df['subclass'] = le.fit_transform(df['subclass'])
    
    # Save label encoder for later use if needed (optional)
    # class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # print(f"Class mapping: {class_mapping}")

    # Handling Outliers (Simple IQR for demonstration)
    features = ['u', 'g', 'r', 'i', 'z', 'redshift']
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Cap outliers instead of removing to preserve data size in small example
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    X = df[features]
    y = df['subclass']

    # 3. Feature Selection (SelectKBest)
    # Since we only have 6 features in synthetic data, we'll keep all 6 or select 5 best
    print("Selecting features...")
    k = min(5, len(features))
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    
    # Get selected feature names
    mask = selector.get_support()
    selected_features = X.columns[mask]
    print(f"Selected Features: {selected_features.tolist()}")

    # 4. Balancing (SMOTE)
    print("Balancing data with SMOTE...")
    # SMOTE requires at least k_neighbors samples, default is 5. 
    # Our synthetic dataset is small (20 rows), so we might need to adjust k_neighbors if classes are very small.
    # Check class distribution
    print(y.value_counts())
    
    try:
        smote = SMOTE(k_neighbors=1, random_state=42) # reduced k_neighbors for small data
        X_resampled, y_resampled = smote.fit_resample(X_new, y)
    except Exception as e:
        print(f"SMOTE failed (likely too little data): {e}. Proceeding without SMOTE.")
        X_resampled, y_resampled = X_new, y

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. Model Building & Training
    print("Training models...")
    
    # Decision Tree
    dtc = DecisionTreeClassifier(random_state=42)
    dtc.fit(X_train, y_train)
    dtc_pred = dtc.predict(X_test)
    dtc_acc = accuracy_score(y_test, dtc_pred)
    print(f"Decision Tree Accuracy: {dtc_acc:.4f}")

    # Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

    # Random Forest
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    rfc_acc = accuracy_score(y_test, rfc_pred)
    print(f"Random Forest Accuracy: {rfc_acc:.4f}")

    # Select Best Model
    best_acc = max(dtc_acc, lr_acc, rfc_acc)
    if best_acc == rfc_acc:
        best_model = rfc
        print("Saving Random Forest as best model.")
    elif best_acc == dtc_acc:
        best_model = dtc
        print("Saving Decision Tree as best model.")
    else:
        best_model = lr
        print("Saving Logistic Regression as best model.")

    # 6. Save Model
    # We also need to save the scaler and likely the selected feature indices if we were strictly following production pipelines,
    # but for this specific request, the user only asked for "kmodel.pkl".
    # Typically, you'd save a pipeline. To keep it simple and consistent with the user's "kmodel.pkl", 
    # we'll assume the input to the app will be raw numbers, so we might need to handle scaling inside the app or save the scaler too.
    # For now, let's just save the model. *Crucial Note*: The app needs to apply the same scaling!
    # To fix this properly, let's bundle them.
    
    model_bundle = {
        'model': best_model,
        'scaler': scaler,
        'selected_features': selected_features.tolist(),
        'label_encoder': le
    }

    with open('kmodel.pkl', 'wb') as f:
        pickle.dump(model_bundle, f)
    print("Model saved to kmodel.pkl")

if __name__ == "__main__":
    train_model()
