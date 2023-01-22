import json
import matplotlib
import numpy
import torch
from captum.attr import LayerIntegratedGradients
from src.expred import (seeding, ExpredInput, BertTokenizerWithSpans, ExpredConfig, Expred)

# Check if the device supports cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configure ExPred or another model
expred_config = ExpredConfig(
    pretrained_dataset_name='fever',
    base_dataset_name='fever',
    device=device,
    load_from_pretrained=True)

# Seeding
seeding(1234)

# Initialize tokenizer
tokenizer = BertTokenizerWithSpans.from_pretrained('bert-base-uncased')

# Create the model
model = Expred.from_pretrained(expred_config)
model.eval()
model = model.to(device)


def read_data(location):
    """
    Reading the data (train or test).

    Args:
        location: The location of the data that needs to be read.

    Returns:
        Returns a list with the lines in the data.
    """
    data = []
    for line in open(location, 'r'):
        data.append(json.loads(line))
    return data


def read_docs(location, name):
    """
    Reading docs that are in the data (txt).

    Args:
        location: The location of the data that needs to be read.
        name: The name of the docs.

    Returns:
        Returns a list with the lines in the docs.
    """
    read_docs = []
    for line in open(location + name, 'r'):
        read_docs.append(line)
    return read_docs


def in_out_put(input_queries, input_docs, input_labels):
    """
    From in- to output with the ExPred model.

    Args:
        input_queries: The queries from the data.
        input_docs: The docs from the data.
        input_labels: The label from the data.

    Returns:
        The input and output from the ExPred model.
    """
    in_out_put_expred_input = ExpredInput(
        queries=[q.split() for q in input_queries],
        docs=[[d.split() for d in input_docs]],
        labels=input_labels,
        config=expred_config,
        ann_ids=['spontan_1'],
        span_tokenizer=tokenizer)

    # Don't forget to preprocess
    in_out_put_expred_input.preprocess()

    # The output is in the form of a dict:
    in_out_put_expred_output = model(in_out_put_expred_input)

    return in_out_put_expred_input, in_out_put_expred_output


def to_tensor(to_tensor_input):
    """
    Encode the input, for Captum.

    Args:
        to_tensor_input: The input from the data.

    Returns:
        The tensor made from the input.
    """
    input_ids_input = [tokenizer.encode(tokenizer.tokenize(t_i),
                                        truncation=True, max_length=512) for t_i in to_tensor_input]
    input_ids = []
    for id_inp in input_ids_input:
        id_inp.remove(101)  # Remove [CLS]
        id_inp.remove(102)  # Remove [SEP]
        input_ids += id_inp
    return torch.tensor(input_ids, device=device)


def wrapper(wrapper_tensor):
    """
    A wrapper for the feature attribution method.

    Args:
        wrapper_tensor: The whole input in tensor form.

    Returns:
        A tensor with the predictions.
    """
    output = model.cls_module.forward(wrapper_tensor.to(torch.int64))  # The output
    return output.get('cls_pred')


def ig(ig_tensor, target, n_steps):
    """
    The Layer Integrated Gradients is defined here (from the library Captum).

    Args:
        ig_tensor: The whole input in tensor form.
        target: The index of the highest score of the predictions.
        n_samples: The amount of iterations or samples that the feature attribution method will iterate over.

    Returns:
        A list with the scores for the tokens.
    """
    ig = LayerIntegratedGradients(wrapper, model.cls_module.bare_bert.embeddings)
    attributions = ig.attribute(ig_tensor.unsqueeze(0),
                                baselines=torch.zeros(len(ig_tensor), device=device).unsqueeze(0),
                                target=target,
                                n_steps=n_steps)
    return attributions


def remove_hash(combined_tokens, fa):
    """
    This method removes the hashtags.

    Args:
        combined_tokens: The tokens.
        fa: The scores from the feature attribution methods.

    Returns:
        Two lists, where the hashtags are removed.
    """
    result_tokens = []
    result_scores = []
    index = 0
    index_result = 0
    while index < len(fa):
        if combined_tokens[index].startswith("##"):
            prev_token = result_tokens[index_result - 1]
            prev_score = result_scores[index_result - 1]
            result_tokens[index_result - 1] = prev_token + combined_tokens[index][2:]
            result_scores[index_result - 1] = max((prev_score, fa[index]), key=abs)
            index += 1
        else:
            result_tokens.append(combined_tokens[index])
            result_scores.append(fa[index])
            index += 1
            index_result += 1
    return result_tokens, result_scores


def marked_sentence(c_tokens, fa, name):
    """
    This method marks/colorizes the sentence (a heatmap is created).
    Source: Source: https://stackoverflow.com/questions/59220488/to-visualize-attention-color-tokens-using-attention-weights

    Args:
        c_tokens: The tokens.
        fa: The scores from the feature attribution methods.
        name: The name of the .txt file that is created at the end.
    """
    colors_positive = matplotlib.cm.Greens
    colors_neutral = matplotlib.cm.Greys
    colors_negative = matplotlib.cm.Reds
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for token, score in zip(c_tokens, fa):
        if score < 0:
            score = matplotlib.colors.rgb2hex(colors_negative(-score)[:3])
        elif score == 0:
            score = matplotlib.colors.rgb2hex(colors_neutral(score)[:3])
        else:
            score = matplotlib.colors.rgb2hex(colors_positive(score)[:3])
        colored_string += template.format(score, '&nbsp' + token + '&nbsp')

    with open('/dir/to/your/colorized_integrated_gradients_both_' + name + '.html', 'w') as f:
        f.write('<p style="word-wrap: break-word; word-break: break-all;">' + colored_string + '</p>')


if __name__ == '__main__':
    # Read from the data, specify the location for the data and the docs
    read_data = read_data('/dir/to/your/{test, train}.jsonl')
    location_docs = '/dir/to/your/docs/'
    # The number of the claim, and the amount of iterations for the feature attribution method
    num = 0  # For when you want to run a specific instance
    # num = int(sys.argv[1])  # For when you want to use run.py
    iterations = 50

    with open("/dir/to/your/integrated_gradients_both_" + str(num) + ".json", "w") as f:
        # Get the queries, docs and label
        queries = [read_data[num].get('query')]
        docs = read_docs(location_docs, read_data[num].get('docids')[0])
        labels = [read_data[num].get('classification')]
        tensor_queries = to_tensor(queries)
        tensor_docs = to_tensor(docs)
        complete_input_tensor = torch.cat((torch.tensor([tokenizer.cls_token_id]), tensor_queries,
                                           torch.tensor([tokenizer.sep_token_id]), tensor_docs,
                                           torch.tensor([tokenizer.sep_token_id])))

        # In-out-put with the explanation and classification of ExPred
        expred_input, expred_output = in_out_put(queries, docs, labels)

        # Integrated Gradients
        # First the selector part of the model
        explanation = expred_output.get('mtl_preds').get('exp_preds').data[0].detach().numpy()
        pred = [round(i) for i in explanation]
        selected_tensor_docs = []
        complete_input = complete_input_tensor.detach().numpy()
        for index in range(len(tensor_queries) + 2, len(tensor_queries) + len(tensor_docs) + 2):
            if index < 511 and pred[index] == 1:  # 511 (+ 1 seperator token) is the max for ExPred
                selected_tensor_docs.append(complete_input[index])
            if index < 511 and pred[index] == 0:  # 511 (+ 1 seperator token) is the max for ExPred
                selected_tensor_docs.append(1012)  # '.', the wildcard fom ExPred
        selected_tensor_docs = torch.tensor(selected_tensor_docs)
        # Second part of the model, predictor stage with IG
        complete_selected_input_tensor = torch.cat((torch.tensor([tokenizer.cls_token_id]), tensor_queries,
                                                    torch.tensor([tokenizer.sep_token_id]), selected_tensor_docs,
                                                    torch.tensor([tokenizer.sep_token_id])))
        classification = expred_output.get('cls_preds').get('cls_pred').detach().numpy()
        attribution = ig(complete_selected_input_tensor, int(numpy.argmax(classification)), iterations)
        attribution = attribution.data[0].sum(dim=1).squeeze(0)
        attribution = attribution.detach().numpy() / numpy.linalg.norm(attribution.detach().numpy())  # Normalize

        # Visualize
        combine_tokens = []
        for i in range(min(511, len(complete_input_tensor) - 1)):  # 511 (+ 1 seperator token) is the max for ExPred
            combine_tokens.append(tokenizer.decode(complete_input_tensor.detach().numpy()[i]).replace(' ', ''))
        combine_tokens.append('[SEP]')
        result_tokens, result_scores = remove_hash(combine_tokens, attribution)
        marked_sentence(result_tokens, result_scores, 'normalized_without_hashtags_' + str(num))

        # Write to json
        json.dump({
            "amount_of_iterations": iterations,
            "tokens_list": result_tokens,
            "scores_list": list(map(lambda x: float(x), result_scores))
        }, f)
