import pandas as pd
import great_expectations as gx
import great_expectations.expectations as gxe


def validate_telco_data(df: pd.DataFrame):
    """
    Validates raw Telco churn data with Great Expectations.

    Returns:
        (is_valid, failed): bool and a list of failed expectation descriptions.
    """
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("telco")
    asset = source.add_dataframe_asset("telco_df")
    batch_def = asset.add_batch_definition_whole_dataframe("whole_df")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    required = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "InternetService", "Contract", "PaymentMethod",
        "MonthlyCharges", "TotalCharges", "Churn",
    ]

    expectations = [gxe.ExpectColumnToExist(column=c) for c in required]
    expectations += [
        gxe.ExpectColumnValuesToNotBeNull(column="Churn"),
        gxe.ExpectColumnValuesToBeInSet(column="Churn", value_set=["Yes", "No"]),
        gxe.ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"]),
        gxe.ExpectColumnValuesToBeInSet(column="SeniorCitizen", value_set=[0, 1]),
        gxe.ExpectColumnValuesToBeBetween(column="tenure", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0),
    ]

    failed = []
    for exp in expectations:
        result = batch.validate(exp)
        if not result.success:
            failed.append(f"{type(exp).__name__}({exp.column})")

    return len(failed) == 0, failed
