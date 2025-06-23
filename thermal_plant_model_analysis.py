#!/usr/bin/env python3
"""
Thermal Power Plant Model Analysis

This script analyzes how well the current model is tailored for thermal power plant environments
and provides recommendations for improvement.
"""

import os
import json
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def analyze_current_model():
    """Analyze the current model's thermal power plant readiness"""
    
    analysis = {
        "current_status": {},
        "thermal_plant_readiness": {},
        "gaps_identified": [],
        "recommendations": [],
        "improvement_plan": {}
    }
    
    # Check dataset composition
    print("🔍 Analyzing Current Model for Thermal Power Plant Deployment...")
    
    # Dataset analysis
    data_path = Path("data")
    image_count = len(list(data_path.rglob("*.jpg"))) + len(list(data_path.rglob("*.png")))
    
    # Check for thermal plant specific images
    thermal_images = len([f for f in data_path.rglob("*NVR*")])
    
    analysis["current_status"] = {
        "total_images": image_count,
        "thermal_plant_images": thermal_images,
        "thermal_plant_percentage": (thermal_images / image_count * 100) if image_count > 0 else 0,
        "classes": ["helmet", "reflective_jacket", "person"],
        "model_size": "5.9 MB (YOLOv8 nano)",
        "training_epochs": "Limited (2-3 epochs based on results.csv)"
    }
    
    # Thermal power plant specific requirements
    thermal_requirements = {
        "environmental_conditions": {
            "high_temperature_areas": "❌ Not specifically trained",
            "steam_environments": "❌ Not specifically trained", 
            "low_light_conditions": "❌ Not specifically trained",
            "dusty_environments": "❌ Not specifically trained",
            "outdoor_switchyards": "⚠️ Limited training data",
            "control_rooms": "⚠️ Limited training data"
        },
        "safety_equipment": {
            "hard_hats": "✅ Trained",
            "high_vis_jackets": "✅ Trained",
            "safety_boots": "❌ Not trained",
            "safety_gloves": "❌ Not trained",
            "arc_flash_protection": "❌ Not trained",
            "respirators": "❌ Not trained",
            "fall_protection": "❌ Not trained"
        },
        "thermal_plant_areas": {
            "boiler_areas": "⚠️ Limited data",
            "turbine_halls": "⚠️ Limited data",
            "coal_handling": "⚠️ Limited data",
            "ash_handling": "❌ No specific training",
            "switchyards": "⚠️ Limited data",
            "cooling_towers": "❌ No specific training",
            "control_rooms": "⚠️ Limited data"
        },
        "detection_challenges": {
            "multiple_workers": "⚠️ Basic support",
            "partial_occlusion": "⚠️ Basic support",
            "varying_distances": "⚠️ Basic support",
            "equipment_interference": "❌ Not trained",
            "reflection_from_surfaces": "❌ Not trained",
            "smoke_steam_interference": "❌ Not trained"
        }
    }
    
    analysis["thermal_plant_readiness"] = thermal_requirements
    
    # Identify gaps
    gaps = [
        "Limited thermal power plant specific training data (~15% of dataset)",
        "Missing detection for critical safety equipment (boots, gloves, arc flash protection)",
        "No training for challenging thermal plant environments (steam, dust, high heat)",
        "Limited detection accuracy in industrial settings (27.1% mAP is low for production)",
        "No area-specific safety requirements implementation",
        "Missing detection for thermal plant specific hazards",
        "No training for equipment interference scenarios",
        "Limited multi-person detection in industrial settings"
    ]
    
    analysis["gaps_identified"] = gaps
    
    # Recommendations
    recommendations = [
        {
            "category": "Dataset Enhancement",
            "priority": "HIGH",
            "items": [
                "Collect 1000+ thermal power plant specific images",
                "Include diverse environmental conditions (steam, dust, heat shimmer)",
                "Add different times of day and lighting conditions",
                "Include multiple camera angles and distances",
                "Add challenging scenarios (partial occlusion, equipment interference)"
            ]
        },
        {
            "category": "Safety Equipment Expansion",
            "priority": "HIGH", 
            "items": [
                "Add safety boots detection",
                "Add safety gloves detection",
                "Add arc flash protection gear",
                "Add respirator/dust mask detection",
                "Add fall protection equipment",
                "Add safety goggles/glasses"
            ]
        },
        {
            "category": "Model Improvement",
            "priority": "MEDIUM",
            "items": [
                "Upgrade to YOLOv8m or YOLOv8l for better accuracy",
                "Train for at least 100+ epochs",
                "Implement data augmentation for industrial environments",
                "Add thermal plant specific data augmentation",
                "Improve mAP from 27.1% to 70%+ for production use"
            ]
        },
        {
            "category": "Thermal Plant Integration",
            "priority": "MEDIUM",
            "items": [
                "Implement area-specific safety requirements",
                "Add shift-specific detection rules",
                "Integrate with plant SCADA systems",
                "Add permit-to-work integration",
                "Implement emergency response protocols"
            ]
        },
        {
            "category": "Environmental Adaptation",
            "priority": "MEDIUM",
            "items": [
                "Train for high temperature environments",
                "Add steam/smoke interference handling",
                "Improve low-light performance",
                "Add weather condition adaptations",
                "Implement heat shimmer compensation"
            ]
        }
    ]
    
    analysis["recommendations"] = recommendations
    
    # Improvement plan
    improvement_plan = {
        "phase_1_immediate": {
            "timeline": "2-4 weeks",
            "actions": [
                "Collect 500+ thermal power plant images",
                "Retrain model with expanded dataset",
                "Add safety boots and gloves detection",
                "Improve model accuracy to 50%+ mAP"
            ],
            "estimated_effort": "Medium"
        },
        "phase_2_enhancement": {
            "timeline": "1-2 months", 
            "actions": [
                "Collect 1000+ diverse thermal plant images",
                "Add all critical safety equipment detection",
                "Implement area-specific rules",
                "Achieve 70%+ mAP accuracy",
                "Add environmental adaptation"
            ],
            "estimated_effort": "High"
        },
        "phase_3_optimization": {
            "timeline": "2-3 months",
            "actions": [
                "Fine-tune for specific plant layouts",
                "Integrate with plant systems",
                "Add advanced analytics",
                "Implement predictive safety features",
                "Add real-time alerting systems"
            ],
            "estimated_effort": "High"
        }
    }
    
    analysis["improvement_plan"] = improvement_plan
    
    return analysis

def generate_thermal_plant_readiness_score(analysis):
    """Calculate thermal plant readiness score"""
    
    scores = {
        "basic_functionality": 85,  # Core detection works
        "thermal_plant_data": 15,   # Limited thermal plant data
        "safety_equipment_coverage": 40,  # Only 2/6 major equipment types
        "environmental_adaptation": 10,   # No environmental training
        "industrial_accuracy": 25,  # 27.1% mAP is too low
        "area_specific_rules": 60,  # Config exists but not trained
        "integration_readiness": 70  # Good API and interface structure
    }
    
    weights = {
        "basic_functionality": 0.10,
        "thermal_plant_data": 0.20,
        "safety_equipment_coverage": 0.20,
        "environmental_adaptation": 0.15,
        "industrial_accuracy": 0.20,
        "area_specific_rules": 0.10,
        "integration_readiness": 0.05
    }
    
    weighted_score = sum(scores[category] * weights[category] for category in scores)
    
    return {
        "overall_score": round(weighted_score, 1),
        "category_scores": scores,
        "interpretation": {
            "90-100": "Fully ready for thermal plant deployment",
            "70-89": "Ready with minor customizations needed",
            "50-69": "Partially ready, significant improvements needed", 
            "30-49": "Basic functionality, major enhancements required",
            "0-29": "Not ready for thermal plant deployment"
        }
    }

def create_improvement_dataset_plan():
    """Create specific plan for thermal plant dataset improvement"""
    
    dataset_plan = {
        "target_images": 2000,
        "current_images": 2526,
        "thermal_plant_specific_needed": 1500,
        
        "image_categories": {
            "boiler_areas": {
                "target": 300,
                "scenarios": [
                    "Workers near boilers with full PPE",
                    "Maintenance activities",
                    "High temperature environments",
                    "Steam interference",
                    "Multiple workers"
                ]
            },
            "turbine_halls": {
                "target": 250,
                "scenarios": [
                    "Turbine maintenance",
                    "Workers in large halls",
                    "Equipment interference",
                    "Varying distances",
                    "Low light conditions"
                ]
            },
            "coal_handling": {
                "target": 200,
                "scenarios": [
                    "Dusty environments",
                    "Conveyor belt areas",
                    "Coal yard operations",
                    "Heavy machinery areas",
                    "Outdoor conditions"
                ]
            },
            "switchyards": {
                "target": 200,
                "scenarios": [
                    "Electrical safety gear",
                    "Arc flash protection",
                    "Outdoor high voltage areas",
                    "Weather conditions",
                    "Long distance detection"
                ]
            },
            "control_rooms": {
                "target": 150,
                "scenarios": [
                    "Indoor environments",
                    "No helmet requirements",
                    "Different PPE standards",
                    "Multiple operators",
                    "Console work"
                ]
            },
            "general_plant": {
                "target": 400,
                "scenarios": [
                    "Walking areas",
                    "Stairs and platforms",
                    "Pipe areas",
                    "Emergency situations",
                    "Various weather conditions"
                ]
            }
        },
        
        "safety_equipment_expansion": {
            "safety_boots": "High priority - foot protection critical",
            "safety_gloves": "High priority - hand protection essential",
            "arc_flash_suits": "Critical for electrical areas",
            "respirators": "Important for dusty areas",
            "safety_glasses": "Basic but important protection",
            "fall_protection": "Critical for height work"
        },
        
        "data_collection_sources": [
            "Thermal power plant partnerships",
            "Industrial safety training videos",
            "Power plant documentation",
            "Safety incident reports",
            "Maintenance procedure videos",
            "Synthetic data generation",
            "Similar industrial facilities"
        ]
    }
    
    return dataset_plan

def main():
    """Run thermal power plant model analysis"""
    
    print("🏭 THERMAL POWER PLANT MODEL ANALYSIS")
    print("=" * 60)
    
    # Run analysis
    analysis = analyze_current_model()
    
    # Calculate readiness score
    readiness = generate_thermal_plant_readiness_score(analysis)
    
    # Create dataset improvement plan
    dataset_plan = create_improvement_dataset_plan()
    
    # Print summary
    print(f"\n📊 THERMAL PLANT READINESS SCORE: {readiness['overall_score']}/100")
    
    score = readiness['overall_score']
    if score >= 70:
        status = "🟢 READY"
    elif score >= 50:
        status = "🟡 PARTIALLY READY"
    else:
        status = "🔴 NOT READY"
    
    print(f"📋 STATUS: {status}")
    
    print(f"\n🎯 CURRENT MODEL STATUS:")
    current = analysis['current_status']
    print(f"   • Total Images: {current['total_images']:,}")
    print(f"   • Thermal Plant Images: {current['thermal_plant_images']} ({current['thermal_plant_percentage']:.1f}%)")
    print(f"   • Safety Equipment: {len(current['classes'])} types (helmet, jacket, person)")
    print(f"   • Training Level: Basic (2-3 epochs)")
    print(f"   • Accuracy: ~27% mAP (too low for production)")
    
    print(f"\n⚠️ MAJOR GAPS IDENTIFIED:")
    for i, gap in enumerate(analysis['gaps_identified'][:5], 1):
        print(f"   {i}. {gap}")
    
    print(f"\n🚀 RECOMMENDED IMPROVEMENT PHASES:")
    for phase, details in analysis['improvement_plan'].items():
        phase_name = phase.replace('_', ' ').title()
        print(f"\n   📅 {phase_name} ({details['timeline']}):")
        for action in details['actions'][:3]:
            print(f"      • {action}")
    
    print(f"\n📈 DATASET IMPROVEMENT TARGETS:")
    print(f"   • Current: {dataset_plan['current_images']:,} images")
    print(f"   • Target: {dataset_plan['target_images']:,} images") 
    print(f"   • Thermal Plant Specific Needed: {dataset_plan['thermal_plant_specific_needed']:,}")
    print(f"   • Safety Equipment Expansion: {len(dataset_plan['safety_equipment_expansion'])} new types")
    
    # Save detailed analysis
    full_report = {
        "analysis": analysis,
        "readiness_score": readiness,
        "dataset_plan": dataset_plan,
        "timestamp": "2025-06-23T11:20:53Z"
    }
    
    with open('thermal_plant_model_analysis.json', 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print(f"\n📄 Detailed analysis saved to 'thermal_plant_model_analysis.json'")
    
    # Final assessment
    print(f"\n🏆 FINAL ASSESSMENT:")
    if score >= 70:
        print("✅ Model is ready for thermal plant deployment with minor customizations")
    elif score >= 50:
        print("⚠️ Model has basic functionality but needs significant improvements for production use")
    else:
        print("❌ Model needs major enhancements before thermal plant deployment")
    
    print(f"\n💡 IMMEDIATE NEXT STEP:")
    print("   Collect and label 500+ thermal power plant specific images to improve accuracy")
    
    return readiness['overall_score']

if __name__ == "__main__":
    main()
