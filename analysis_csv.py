import pandas as pd
import argparse #  argparse 
import rich

def analyze_csv(filepath):
    """
    CSV

    Args:
        filepath (str): CSV
    """
    try:
        df = pd.read_csv(filepath, engine='python')
        print(f": '{filepath}'")
        print("-" * 30)

        # float64  float32
        float_columns = df.select_dtypes(include=['float64', 'float32'])

        if float_columns.empty:
            print("")
            return

        print("")
        
        # 
        for col in float_columns.columns:
            
            EXCLUDE_WORDS = ["bmshj2018","cheng200","all_no"]
            STRICT_EXCLUDE_WORDS = ["No",]
            
            continue_flag = False
            for exclude_word in EXCLUDE_WORDS:
                if exclude_word in col:
                    continue_flag = True
            for exclude_word in STRICT_EXCLUDE_WORDS:
                if exclude_word == col:
                    continue_flag = True
            if continue_flag:
                continue
            
            # 
            mean_val = float_columns[col].mean()
            #  (ddof=1)
            variance_val = float_columns[col].var()
            
            rich.print(f"{col:>30}  Mean: {mean_val:.4f}, Var: {variance_val:.4f}")

    except FileNotFoundError:
        print(f":  '{filepath}' ")
    except pd.errors.EmptyDataError:
        print(f":  '{filepath}' ")
    except Exception as e:
        print(f": {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSV"
    )
    parser.add_argument(
        "filepath", 
        type=str, 
        help="CSV"
    )

    args = parser.parse_args()
    analyze_csv(args.filepath)
