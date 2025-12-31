#!/usr/bin/env python3

"""
fuzz_test_generator.py
Generate random fuzz tests for NeuronDB crash testing.

Creates SQL test cases with random/malformed inputs to find edge cases.
"""

import argparse
import json
import os
import random
import sys
from typing import List, Dict


class FuzzTestGenerator:
    """Generate fuzz test cases."""
    
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        self.tests = []
    
    def generate_vector_literal(self, dim: int = None, malformed: bool = False) -> str:
        """Generate a vector literal, possibly malformed."""
        if malformed:
            malformed_types = [
                "[1,2,3",  # Missing closing bracket
                "1,2,3]",  # Missing opening bracket
                "[1,2,3,]",  # Trailing comma
                "[1,2,three]",  # Non-numeric
                "vector '[NaN,Infinity,-Infinity]'",  # Special values
                "vector '[]'::vector",  # Empty
                "NULL",  # NULL
                "vector '[1e308,1e308,1e308]'",  # Overflow
            ]
            return random.choice(malformed_types)
        
        if dim is None:
            dim = random.randint(1, 1536)
        
        values = [random.uniform(-100, 100) for _ in range(dim)]
        return f"vector '[{','.join(str(v) for v in values)}]'"
    
    def generate_jsonb_params(self, malformed: bool = False) -> str:
        """Generate JSONB parameters."""
        if malformed:
            malformed_params = [
                "'{'unclosed'",
                "'{invalid json}'",
                "'{\"key\": undefined}'",
                "NULL",
                "'{}'::text",  # Wrong type
            ]
            return random.choice(malformed_params)
        
        params = {
            "learning_rate": random.uniform(0.001, 1.0),
            "epochs": random.randint(10, 1000),
            "batch_size": random.choice([16, 32, 64, 128]),
        }
        # Randomly add optional params
        if random.random() < 0.5:
            params["regularization"] = random.uniform(0.0, 0.1)
        
        json_str = json.dumps(params)
        return f"'{json_str}'::jsonb"
    
    def generate_table_name(self) -> str:
        """Generate a table name (may not exist)."""
        prefixes = ["test", "temp", "public", "neurondb"]
        suffixes = ["_table", "_data", "_train", "_test", ""]
        name = random.choice(prefixes) + random.choice(suffixes) + str(random.randint(1, 999))
        return name
    
    def generate_column_name(self) -> str:
        """Generate a column name."""
        names = ["features", "vectors", "embedding", "data", "x", "y", "label", "target"]
        return random.choice(names)
    
    def generate_train_test(self) -> str:
        """Generate a train function test."""
        algorithms = [
            "linear_regression", "logistic_regression", "random_forest",
            "svm", "kmeans", "neural_network"
        ]
        
        algorithm = random.choice(algorithms)
        table_name = self.generate_table_name()
        feature_col = self.generate_column_name()
        label_col = random.choice(["label", "target", "y"])
        
        # Randomly add malformed inputs
        use_malformed = random.random() < 0.3
        
        if use_malformed:
            params = self.generate_jsonb_params(malformed=True)
        else:
            params = self.generate_jsonb_params(malformed=False)
        
        # Randomly use NULL
        if random.random() < 0.2:
            if random.random() < 0.5:
                table_name = "NULL"
            else:
                feature_col = "NULL"
        
        return f"SELECT neurondb.train('{algorithm}', '{table_name}', '{feature_col}', '{label_col}', {params});"
    
    def generate_predict_test(self) -> str:
        """Generate a predict function test."""
        model_ids = [1, 100, 999999, -1, 0]
        model_id = random.choice(model_ids)
        
        # Random vector dimension
        dim = random.choice([128, 384, 512, 768, 1536, 2048])
        malformed = random.random() < 0.3
        
        vector = self.generate_vector_literal(dim=dim, malformed=malformed)
        
        return f"SELECT neurondb.predict('model_{model_id}', {vector});"
    
    def generate_evaluate_test(self) -> str:
        """Generate an evaluate function test."""
        model_ids = [1, 100, 999999, -1, 0]
        model_id = random.choice(model_ids)
        
        table_name = self.generate_table_name()
        feature_col = self.generate_column_name()
        label_col = random.choice(["label", "target", "y"])
        
        return f"SELECT neurondb.evaluate({model_id}, '{table_name}', '{feature_col}', '{label_col}');"
    
    def generate_vector_operation_test(self) -> str:
        """Generate a vector operation test."""
        operations = [
            ("vector_l2_distance", 2),
            ("vector_cosine_distance", 2),
            ("vector_inner_product", 2),
            ("vector_add", 2),
            ("vector_subtract", 2),
        ]
        
        op_name, num_args = random.choice(operations)
        
        dim = random.choice([128, 384, 512, 768])
        malformed = random.random() < 0.2
        
        args = [self.generate_vector_literal(dim=dim, malformed=malformed) for _ in range(num_args)]
        
        return f"SELECT {op_name}({', '.join(args)});"
    
    def generate_test_batch(self, count: int = 1000) -> List[str]:
        """Generate a batch of fuzz tests."""
        tests = []
        
        test_generators = [
            self.generate_train_test,
            self.generate_predict_test,
            self.generate_evaluate_test,
            self.generate_vector_operation_test,
        ]
        
        for _ in range(count):
            generator = random.choice(test_generators)
            try:
                test = generator()
                tests.append(test)
            except Exception as e:
                # Skip tests that fail to generate
                continue
        
        return tests
    
    def write_test_file(self, tests: List[str], output_file: str):
        """Write tests to a SQL file."""
        with open(output_file, "w") as f:
            f.write("/*\n")
            f.write(" * Generated fuzz tests for NeuronDB crash testing\n")
            f.write(f" * {len(tests)} test cases\n")
            f.write(" */\n\n")
            f.write("\\set ON_ERROR_STOP off\n\n")
            
            for i, test in enumerate(tests, 1):
                f.write(f"-- Test {i}\n")
                f.write(f"{test}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Generate fuzz tests for NeuronDB")
    parser.add_argument("--count", type=int, default=1000, help="Number of tests to generate")
    parser.add_argument("--output", default="fuzz_tests.sql", help="Output SQL file")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    generator = FuzzTestGenerator(seed=args.seed)
    tests = generator.generate_test_batch(args.count)
    generator.write_test_file(tests, args.output)
    
    print(f"Generated {len(tests)} fuzz tests")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()


