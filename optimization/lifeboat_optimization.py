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

class LifeboatOptimizer:
    """
    Optimize lifeboat seat allocation using predicted survival probabilities
    """
    
    def __init__(self, model_path='../model.pkl', scaler_path='../scaler.pkl',
                 le_sex_path='../le_sex.pkl', le_embarked_path='../le_embarked.pkl'):
        """Load ML model and preprocessors"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.le_sex = joblib.load(le_sex_path)
        self.le_embarked = joblib.load(le_embarked_path)
    
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
        plt.show()

def main():
    """Example usage"""
    print("="*70)
    print("TITANIC LIFEBOAT RESOURCE ALLOCATION - OPTIMIZATION")
    print("="*70)
    
    # Load sample data
    print("\nLoading Titanic dataset...")
    df = pd.read_csv('../data/titanic.csv')
    
    # Take a subset for demonstration
    sample_df = df.sample(n=200, random_state=42).reset_index(drop=True)
    
    # Initialize optimizer
    print("Initializing optimizer...")
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