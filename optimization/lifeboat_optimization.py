#!/usr/bin/env python3
"""
Titanic Lifeboat Resource Allocation - Operations Research Optimization

This module uses linear programming to optimize lifeboat seat allocation
based on predicted survival probabilities from the ML model.
"""

import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import logging
import os
from datetime import datetime
import time
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables for configuration
S3_BUCKET = os.getenv('S3_BUCKET_NAME', 'titanic-prediction-bucket')
DYNAMODB_TABLE_OPTIMIZATION = os.getenv('DYNAMODB_TABLE_OPTIMIZATION', 'optimization-results')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Lazy AWS client initialization
_s3_client = None
_dynamodb_client = None

def get_s3_client():
    """Lazy initialization of S3 client"""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3', region_name=AWS_REGION)
    return _s3_client

def get_dynamodb_client():
    """Lazy initialization of DynamoDB client"""
    global _dynamodb_client
    if _dynamodb_client is None:
        _dynamodb_client = boto3.client('dynamodb', region_name=AWS_REGION)
    return _dynamodb_client

class LifeboatOptimizer:
    """
    Optimize lifeboat seat allocation using predicted survival probabilities
    """
    
    def __init__(self):
        """Load ML model and preprocessors from S3 or locally"""
        UPLOAD_TO_S3 = os.getenv('UPLOAD_TO_S3', 'false').lower() == 'true'
        try:
            if UPLOAD_TO_S3:
                s3 = get_s3_client()
                # Use BytesIO for streaming load to avoid loading entire file into memory
                self.model = joblib.load(BytesIO(s3.get_object(Bucket=S3_BUCKET, Key='model.pkl')['Body'].read()))
                self.scaler = joblib.load(BytesIO(s3.get_object(Bucket=S3_BUCKET, Key='scaler.pkl')['Body'].read()))
                self.le_sex = joblib.load(BytesIO(s3.get_object(Bucket=S3_BUCKET, Key='le_sex.pkl')['Body'].read()))
                self.le_embarked = joblib.load(BytesIO(s3.get_object(Bucket=S3_BUCKET, Key='le_embarked.pkl')['Body'].read()))
                logger.info("Models loaded successfully from S3 for optimization")
            else:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.model = joblib.load(os.path.join(base_dir, 'model.pkl'))
                self.scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
                self.le_sex = joblib.load(os.path.join(base_dir, 'le_sex.pkl'))
                self.le_embarked = joblib.load(os.path.join(base_dir, 'le_embarked.pkl'))
                logger.info("Models loaded successfully from local files for optimization")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def log_optimization_to_dynamodb(self, results, capacity, priority_children, priority_women, max_family_members):
        """Log optimization results to DynamoDB"""
        try:
            dynamodb = get_dynamodb_client()
            item = {
                'id': {'S': str(int(time.time() * 1000000))},  # Unique ID
                'timestamp': {'S': datetime.utcnow().isoformat()},
                'capacity': {'N': str(capacity)},
                'status': {'S': results['status']},
                'objective_value': {'N': str(results['objective_value'])},
                'selected_count': {'N': str(results['selected_count'])},
                'utilization': {'N': str(results['utilization'])},
                'priority_children': {'BOOL': priority_children},
                'priority_women': {'BOOL': priority_women},
                'max_family_members': {'N': str(max_family_members) if max_family_members else '0'}
            }
            dynamodb.put_item(TableName=DYNAMODB_TABLE_OPTIMIZATION, Item=item)
            logger.info("Optimization results logged to DynamoDB")
        except Exception as e:
            logger.error(f"Failed to log optimization to DynamoDB: {e}")

    def predict_survival_probabilities(self, passenger_df):
        """
        Predict survival probabilities for passengers
        
        Args:
            passenger_df: DataFrame with passenger features
            
        Returns:
            Array of survival probabilities
        """
        # Feature engineering
        passenger_df['family_size'] = passenger_df['sibsp'] + passenger_df['parch'] + 1
        passenger_df['is_alone'] = (passenger_df['family_size'] == 1).astype(int)
        
        # Select features
        features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 
                   'embarked', 'family_size', 'is_alone']
        X = passenger_df[features].copy()
        
        # Encode
        X['sex'] = self.le_sex.transform(X['sex'])
        X['embarked'] = self.le_embarked.transform(X['embarked'])
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return probabilities
    
    def optimize_allocation(self, passengers_df, capacity, 
                          priority_children=True, 
                          priority_women=True,
                          max_family_members=None):
        """
        Solve lifeboat allocation optimization problem
        
        Args:
            passengers_df: DataFrame with passenger data
            capacity: Number of available lifeboat seats
            priority_children: Give priority to children (age < 18)
            priority_women: Give priority to women
            max_family_members: Max family members allowed per family
            
        Returns:
            Dictionary with optimization results
        """
        
        # Get survival probabilities
        probabilities = self.predict_survival_probabilities(passengers_df)
        passengers_df['survival_prob'] = probabilities
        
        n = len(passengers_df)
        
        # Create optimization problem
        problem = LpProblem("Lifeboat_Allocation", LpMaximize)
        
        # Decision variables: x[i] = 1 if passenger i gets a seat
        x = [LpVariable(f"x_{i}", cat='Binary') for i in range(n)]
        
        # Objective: Maximize expected survivors
        problem += lpSum([probabilities[i] * x[i] for i in range(n)]), "Total_Expected_Survivors"
        
        # Constraint 1: Capacity
        problem += lpSum(x) <= capacity, "Seat_Capacity"
        
        # Constraint 2: Priority for children (if enabled)
        if priority_children:
            children_indices = passengers_df[passengers_df['age'] < 18].index.tolist()
            if children_indices:
                # At least 30% of seats for children if available
                min_children = min(len(children_indices), int(0.3 * capacity))
                problem += lpSum([x[i] for i in children_indices]) >= min_children, "Children_Priority"
        
        # Constraint 3: Priority for women (if enabled)
        if priority_women:
            women_indices = passengers_df[passengers_df['sex'] == 'female'].index.tolist()
            if women_indices:
                # At least 50% of seats for women if available
                min_women = min(len(women_indices), int(0.5 * capacity))
                problem += lpSum([x[i] for i in women_indices]) >= min_women, "Women_Priority"
        
        # Constraint 4: Family member limit (optional)
        if max_family_members:
            families = passengers_df.groupby(['sibsp', 'parch']).groups
            for family_key, family_indices in families.items():
                if len(family_indices) > 1:  # Only for actual families
                    problem += lpSum([x[i] for i in family_indices]) <= max_family_members, \
                              f"Family_{family_key}_Limit"
        
        # Solve
        problem.solve()
        
        # Extract results
        selected = [i for i in range(n) if value(x[i]) == 1]
        
        results = {
            'status': LpStatus[problem.status],
            'objective_value': value(problem.objective),
            'selected_passengers': selected,
            'selected_count': len(selected),
            'capacity': capacity,
            'utilization': len(selected) / capacity * 100,
            'passengers_data': passengers_df.iloc[selected].copy()
        }
        
        # Log optimization results
        self.log_optimization_to_dynamodb(results, capacity, priority_children, priority_women, max_family_members)

        return results

    def visualize_results(self, results, passengers_df):
        """Visualize optimization results"""
        
        selected_df = results['passengers_data']
        not_selected_df = passengers_df.drop(results['selected_passengers'])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Lifeboat Allocation Optimization Results', fontsize=16, fontweight='bold')
        
        # 1. Selection by class
        class_counts = selected_df['pclass'].value_counts().sort_index()
        axes[0, 0].bar(class_counts.index, class_counts.values, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Selected Passengers by Class')
        axes[0, 0].set_xlabel('Passenger Class')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Selection by gender
        gender_counts = selected_df['sex'].value_counts()
        axes[0, 1].bar(gender_counts.index, gender_counts.values, color=['pink', 'lightblue'], edgecolor='black')
        axes[0, 1].set_title('Selected Passengers by Gender')
        axes[0, 1].set_xlabel('Gender')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Age distribution
        axes[0, 2].hist(selected_df['age'], bins=20, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Age Distribution of Selected')
        axes[0, 2].set_xlabel('Age')
        axes[0, 2].set_ylabel('Frequency')
        
        # 4. Survival probability comparison
        axes[1, 0].hist([selected_df['survival_prob'], not_selected_df['survival_prob']], 
                       bins=20, label=['Selected', 'Not Selected'], 
                       color=['green', 'red'], alpha=0.6, edgecolor='black')
        axes[1, 0].set_title('Survival Probability Distribution')
        axes[1, 0].set_xlabel('Predicted Survival Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 5. Summary stats
        summary_text = f"""
        Optimization Results:
        
        Status: {results['status']}
        Total Seats: {results['capacity']}
        Seats Used: {results['selected_count']}
        Utilization: {results['utilization']:.1f}%
        
        Expected Survivors: {results['objective_value']:.2f}
        
        Demographics:
        Children (< 18): {len(selected_df[selected_df['age'] < 18])}
        Women: {len(selected_df[selected_df['sex'] == 'female'])}
        Men: {len(selected_df[selected_df['sex'] == 'male'])}
        
        Avg Survival Prob: {selected_df['survival_prob'].mean():.3f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center', 
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        # 6. Fare distribution
        axes[1, 2].boxplot([selected_df['fare'], not_selected_df['fare']], 
                          labels=['Selected', 'Not Selected'])
        axes[1, 2].set_title('Fare Distribution')
        axes[1, 2].set_ylabel('Fare')
        
        plt.tight_layout()
        plt.savefig('optimization_results.png')
        logger.info("Optimization visualization saved to optimization_results.png")

def main():
    """Example usage"""
    print("="*70)
    print("TITANIC LIFEBOAT RESOURCE ALLOCATION - OPTIMIZATION")
    print("="*70)
    
    # Load sample data
    print("\nLoading Titanic dataset...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base_dir, 'data', 'titanic.csv'))
    
    # Take a subset for demonstration
    sample_df = df.sample(n=200, random_state=42).reset_index(drop=True)
    
    # ... rest identical until the end ...
    # Initialize optimizer
    logger.info("Initializing optimizer...")
    optimizer = LifeboatOptimizer()
    
    # Run optimization with different capacity scenarios
    capacities = [50, 100, 150]
    
    for capacity in capacities:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {capacity} lifeboat seats available")
        print(f"{'='*70}")
        
        results = optimizer.optimize_allocation(
            sample_df,
            capacity=capacity,
            priority_children=True,
            priority_women=True,
            max_family_members=4
        )
        
        print(f"\nOptimization Status: {results['status']}")
        print(f"Seats Allocated: {results['selected_count']} / {results['capacity']}")
        print(f"Utilization: {results['utilization']:.1f}%")
        print(f"Expected Survivors: {results['objective_value']:.2f}")
        
        # Show demographics
        selected = results['passengers_data']
        print(f"\nDemographics of selected passengers:")
        print(f"  Children (< 18): {len(selected[selected['age'] < 18])}")
        print(f"  Women: {len(selected[selected['sex'] == 'female'])}")
        print(f"  Men: {len(selected[selected['sex'] == 'male'])}")
        print(f"  Class 1: {len(selected[selected['pclass'] == 1])}")
        print(f"  Class 2: {len(selected[selected['pclass'] == 2])}")
        print(f"  Class 3: {len(selected[selected['pclass'] == 3])}")
        
        # Visualize
        if capacity == 100:  # Show detailed viz for middle scenario
            optimizer.visualize_results(results, sample_df)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()