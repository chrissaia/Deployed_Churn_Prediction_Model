from litellm import completion
import os


def explain(input_dict, feature_importance, proba, preds):
    '''

    :param input_dict: The input dictionary containing the un-preprocessed data
    :param proba: from eval
    :param preds: from eval
    :return: the llm explanation
    '''

    prompt = f"""
        You are an expert churn analyst.

        Customer data:
        {input_dict}
        
        Top model features:
        {feature_importance}

        Model output:
        - Churn probability: {proba}
        - Prediction: {"Likely to churn" if preds[0] == 1 else "Not likely to churn"}

        IMPORTANT:
        - The prediction is the ground truth. Do NOT contradict it.
        - If probability is low, explain why the customer is NOT likely to churn.
        - Only use features that are explicitly present in the data.

        Give a concise explanation (3-5 bullet points) explaining 
        which variables provided more of a reason to churn/not churn and why.

        Finally, give a final 1-2 sentence summary of the customer.
        """

    response = completion(
        model="gpt-4o",
        api_key=os.getenv("LITELLM_MASTER_KEY"),
        messages=[{"role": "user", "content": prompt}]
    )

    explanation = response["choices"][0]["message"]["content"]

    return explanation