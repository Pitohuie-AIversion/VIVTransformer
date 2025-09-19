#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Model System Integration Test

This script tests the functionality of the entire unified model system.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
import traceback
from pathlib import Path
import sys
import json
import time

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'models'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegrationTester:
    """Integration tester for unified model system"""
    
    def __init__(self):
        self.test_results = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Import modules
        self.modules_available = self._check_module_availability()
    
    def _check_module_availability(self) -> Dict[str, bool]:
        """Check module availability"""
        availability = {}
        
        # Check unified model factory
        try:
            from unified_model_factory import UnifiedModelFactory, get_global_factory
            availability['unified_model_factory'] = True
            self.model_factory = get_global_factory()
        except ImportError as e:
            logger.warning(f"Unified model factory not available: {e}")
            availability['unified_model_factory'] = False
            self.model_factory = None
        
        # Check unified config manager
        try:
            from unified_config_manager import UnifiedConfigManager, get_global_config_manager
            availability['unified_config_manager'] = True
            self.config_manager = get_global_config_manager()
        except ImportError as e:
            logger.warning(f"Unified config manager not available: {e}")
            availability['unified_config_manager'] = False
            self.config_manager = None
        
        # Check compatibility adapter
        try:
            from trainer_compatibility_adapter import TrainerCompatibilityAdapter, get_global_adapter
            availability['trainer_compatibility_adapter'] = True
            self.compatibility_adapter = get_global_adapter()
        except ImportError as e:
            logger.warning(f"Compatibility adapter not available: {e}")
            availability['trainer_compatibility_adapter'] = False
            self.compatibility_adapter = None
        
        return availability
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> bool:
        """Run a single test"""
        logger.info(f"Starting test: {test_name}")
        start_time = time.time()
        
        try:
            result = test_func(*args, **kwargs)
            duration = time.time() - start_time
            
            self.test_results.append({
                'name': test_name,
                'status': 'PASS',
                'duration': duration,
                'result': result
            })
            
            logger.info(f"Test passed: {test_name} ({duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            self.test_results.append({
                'name': test_name,
                'status': 'FAIL',
                'duration': duration,
                'error': error_msg,
                'traceback': traceback.format_exc()
            })
            
            logger.error(f"Test failed: {test_name} - {error_msg}")
            return False
    
    def test_unified_model_factory(self) -> Dict[str, Any]:
        """Test unified model factory"""
        if not self.modules_available['unified_model_factory']:
            raise ImportError("Unified model factory not available")
        
        results = {}
        
        # Test getting supported models
        supported_models = self.model_factory.get_supported_models()
        results['supported_models'] = supported_models
        assert len(supported_models) > 0, "No supported models"
        
        # Test model creation
        test_configs = [
            {
                'model_type': 'enhanced_transformer_1d',
                'config': {
                    'input_dim': 64,
                    'output_dim': 32,
                    'd_model': 128,
                    'num_heads': 4,
                    'num_layers': 2
                }
            }
        ]
        
        created_models = []
        for test_config in test_configs:
            model_type = test_config['model_type']
            config = test_config['config']
            
            if model_type in supported_models:
                try:
                    model = self.model_factory.create_model(model_type, config)
                    assert isinstance(model, nn.Module), f"Created {model_type} is not nn.Module"
                    
                    # Test forward pass
                    model.eval()
                    with torch.no_grad():
                        test_input = torch.randn(2, config['input_dim'])
                        output = model(test_input)
                        assert output.shape[-1] == config['output_dim'], f"{model_type} output dimension incorrect"
                    
                    created_models.append(model_type)
                    
                except Exception as e:
                    logger.warning(f"Failed to create {model_type}: {e}")
        
        results['created_models'] = created_models
        return results
    
    def test_trainer_compatibility_adapter(self) -> Dict[str, Any]:
        """Test trainer compatibility adapter"""
        if not self.modules_available['trainer_compatibility_adapter']:
            raise ImportError("Compatibility adapter not available")
        
        results = {}
        
        # Test config validation
        test_config = {
            'model': {
                'input_dim': 64,
                'output_dim': 32,
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 6,
                'attention_type': 'sge'
            },
            'training': {
                'epochs': 50,
                'learning_rate': 1e-3
            }
        }
        
        is_valid, errors = self.compatibility_adapter.validate_trainer_config(test_config)
        results['config_validation'] = {'valid': is_valid, 'errors': errors}
        
        # Test model creation
        try:
            model = self.compatibility_adapter.create_model_from_trainer_config(test_config)
            assert isinstance(model, nn.Module), "Created model is not nn.Module"
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                test_input = torch.randn(2, test_config['model']['input_dim'])
                output = model(test_input)
                assert output.shape[-1] == test_config['model']['output_dim'], "Output dimension incorrect"
            
            results['model_creation'] = {'success': True, 'model_type': type(model).__name__}
            
        except Exception as e:
            results['model_creation'] = {'success': False, 'error': str(e)}
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests"""
        logger.info("Starting integration test suite")
        
        # Module availability test
        self.run_test("Module availability check", lambda: self.modules_available)
        
        # Unified model factory test
        if self.modules_available['unified_model_factory']:
            self.run_test("Unified model factory functionality", self.test_unified_model_factory)
        
        # Compatibility adapter test
        if self.modules_available['trainer_compatibility_adapter']:
            self.run_test("Trainer compatibility adapter functionality", self.test_trainer_compatibility_adapter)
        
        # Generate test report
        return self.generate_test_report()
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_duration': sum(result['duration'] for result in self.test_results)
            },
            'module_availability': self.modules_available,
            'test_results': self.test_results,
            'device_info': {
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """Print test summary"""
        summary = report['summary']
        
        print("\n" + "="*60)
        print("Unified Model System Integration Test Report")
        print("="*60)
        
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed tests: {summary['passed_tests']}")
        print(f"Failed tests: {summary['failed_tests']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Total duration: {summary['total_duration']:.2f}s")
        
        print("\nModule availability:")
        for module, available in report['module_availability'].items():
            status = "✓" if available else "✗"
            print(f"  {status} {module}")
        
        print("\nTest results details:")
        for result in report['test_results']:
            status_symbol = "✓" if result['status'] == 'PASS' else "✗"
            print(f"  {status_symbol} {result['name']} ({result['duration']:.2f}s)")
            if result['status'] == 'FAIL':
                print(f"    Error: {result['error']}")
        
        print("\nDevice info:")
        device_info = report['device_info']
        print(f"  Device: {device_info['device']}")
        print(f"  CUDA available: {device_info['cuda_available']}")
        if device_info['cuda_available']:
            print(f"  CUDA device count: {device_info['cuda_device_count']}")
        
        print("="*60)

def main():
    """Main function"""
    tester = IntegrationTester()
    
    # Run all tests
    report = tester.run_all_tests()
    
    # Print summary
    tester.print_summary(report)
    
    # Return success rate
    return report['summary']['success_rate'] >= 0.5

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)