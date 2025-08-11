import warnings
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict, Generator


class SampleGenerator:
    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame, List[List[float]]],
        labels: Union[np.ndarray, List[int], None] = None,
        feature_columns: Union[List[str], None] = None,
        label_column: Union[str, List[str], None] = None,
        seed: int = None
    ):
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

        if isinstance(data, pd.DataFrame):
            if label_column is None or feature_columns is None:
                raise ValueError("You must provide both 'feature_columns' and 'label_column' when using a DataFrame.")
            
            missing_cols = set(feature_columns + ([label_column] if isinstance(label_column, str) else label_column)) - set(data.columns)
            if missing_cols:
                raise ValueError(f"The following columns are missing in the DataFrame: {missing_cols}")
            
            self.data = data[feature_columns].values
            if isinstance(label_column, list):
                self.labels = data[label_column].values
            else:
                self.labels = data[label_column].values

        else:
            self.data = np.array(data)
            if labels is None:
                raise ValueError("Labels must be provided when data is not a DataFrame.")
            self.labels = np.array(labels)

        if len(self.data) != len(self.labels):
            raise ValueError("Data and labels must have the same length.")

        self.unique_classes = (
            np.unique([tuple(lbl) for lbl in self.labels]) if len(self.labels.shape) > 1 else np.unique(self.labels)
        )

        if len(self.labels.shape) > 1:
            # Para labels multidimensionales, convertir cada fila a tupla y contar
            labels_tuples = [tuple(lbl) for lbl in self.labels]
            self.class_counts = {
                cls: labels_tuples.count(cls)
                for cls in self.unique_classes
            }
        else:
            self.class_counts = {
                cls: np.sum(self.labels == cls)
                for cls in self.unique_classes
            }


    def generate(self, num_samples: int = 5) -> Generator[Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]], None, None]:
        
        if num_samples <= 0:
            raise ValueError("num_samples must be greater than 0.")
        if num_samples >= len(self.data):
            raise ValueError("num_samples must be less than the number of instances.")
        
        
        min_class_size = min(self.class_counts.values())
        available_samples_per_class = min_class_size - 1  # -1 because we exclude the target sample
        
        replace = False
        if num_samples > available_samples_per_class:
            print(
                f"Requested {num_samples} samples, but smallest class has only "
                f"{available_samples_per_class} available samples (excluding target). "
                f"Samples will be repeated.",
                UserWarning,
            )
            replace = True
        
        for index in range(len(self.data)):
            expected_label: np.ndarray = self.labels[index]
            sample_to_predict: np.ndarray = self.data[index]

            current_kw_samples: Dict[str, np.ndarray] = {
                f'class_{cls}': np.zeros((num_samples, len(self.data)))
                for cls in self.unique_classes
            }

            for cls in self.unique_classes:
                mask = np.where(self.labels == cls)[0]
                mask = mask[mask != index]  # Excluir el índice actual
                
                sampled_indices = self.rng.choice(mask, size=num_samples, replace=replace)

                # while True:
                #     sampled_indices = self.rng.choice(mask, size=num_samples, replace=replace)
                #     if index not in sampled_indices:
                #         break
                    
                current_kw_samples[f'class_{cls}'] = self.data[sampled_indices]

            yield sample_to_predict, expected_label, current_kw_samples


if __name__ == '__main__':
    df = pd.DataFrame({
        "f1": [1, 3, 5, 7, 9, 11],
        "f2": [2, 4, 6, 8, 10, 12],
        "label": [1, 1, 1, 2, 2, 2]
    })

    gen = SampleGenerator(df, feature_columns=["f1", "f2"], label_column="label")
    for test_x, test_y, kw_samples in gen.generate(num_samples=2):
        print(f'Test instance: {test_x}')
        print(f'Target: {test_y}')
        print(f'know samples: {kw_samples}')
        print('\n')

    print("="*60)

    x_test: np.ndarray = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    )

    y_test: np.ndarray = np.array(
        [1, 1, 1, 2, 2]
    )
    
    gen = SampleGenerator(data=x_test, labels=y_test)


    for test_x, test_y, kw_samples in gen.generate(num_samples=2):
        print(f'Test instance: {test_x}')
        print(f'Target: {test_y}')
        print(f'know samples: {kw_samples}')
        print('\n')
