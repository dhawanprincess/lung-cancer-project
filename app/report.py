def generate_report(age, smoking, chest_pain, result):
    report = f"""
    LUNG CANCER SCREENING REPORT

    Age: {age}
    Smoking: {smoking}
    Chest Pain: {chest_pain}

    Final Diagnosis: {result}

    Note: This is an AI generated preliminary report.
    Please consult a medical professional.
    """

    with open("report.txt","w") as f:
        f.write(report)

    return report
