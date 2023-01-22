import json
import matplotlib
import numpy
import torch
from captum._utils.models import SkLearnLinearRegression
from captum.attr import LimeBase
from src.expred import (seeding, ExpredInput,
                        BertTokenizerWithSpans, ExpredConfig, Expred)


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
    return torch.tensor([input_ids], device=device)


def to_inputs(tensor, input_docs):
    """
    Decode the tensor of the docs for ExPred.

    Args:
        tensor: The docs from the data in tensor form.
        input_docs: The docs from the data.

    Returns:
        A list with the decoded tokens.
    """
    padded = tensor.detach().numpy()[0]
    t_docs = to_tensor(input_docs).detach().numpy()[0]
    encoded = [tokenizer.encode(tokenizer.tokenize(d), truncation=True, max_length=512) for d in input_docs]
    index = 0
    result = []
    for i in range(len(encoded)):
        temp = []
        for j in range(len(encoded[i]) - 2):
            if padded[index] == 0:
                # temp.append(tokenizer.decode(padded[index]).replace(' ', ''))  # The pad token for other models
                temp.append('.')  # '.' the wildcard from ExPred is used to pad/mask tokens
            if padded[index] == 1:
                temp.append(tokenizer.decode(t_docs[index]).replace(' ', ''))
            index += 1
        result.append(temp)
    return result


def similarity_kernel(original_input, perturbed_input, perturbed_interpretable_input, **kwargs):
    # See the tutorial from Captum: https://captum.ai/tutorials/Image_and_Text_Classification_LIME
    distance = 1 - torch.nn.functional.cosine_similarity(original_input.to(torch.float32),
                                                         perturbed_input.to(torch.float32), dim=1)
    return torch.exp(-1 * (distance ** 2) / 2)


def perturb_func(original_input, **kwargs):
    # See the tutorial from Captum: https://captum.ai/tutorials/Image_and_Text_Classification_LIME
    probs = torch.ones_like(original_input) * 0.5
    return torch.bernoulli(probs).long()


def to_interp_transform(curr_sample, original_inp, **kwargs):
    # See the tutorial from Captum: https://captum.ai/tutorials/Image_and_Text_Classification_LIME
    return curr_sample


def wrapper(wrapper_tensor_docs, wrapper_queries=None, wrapper_docs=None, wrapper_labels=None):
    """
    A wrapper for the feature attribution method.

    Args:
        wrapper_tensor_docs: The docs from the data in tensor form.
        wrapper_queries: The queries from the data.
        wrapper_docs: The docs from the data.
        wrapper_labels: The label from the data.

    Returns:
        A tensor with the predictions.
    """
    docs_from_tensor = to_inputs(wrapper_tensor_docs, wrapper_docs)  # The tensor becomes a List with Lists of words
    expred_input_wrapper = ExpredInput(
        queries=[q.split() for q in wrapper_queries],
        docs=[docs_from_tensor],
        labels=wrapper_labels,
        config=expred_config,
        ann_ids=['spontan_1'],
        span_tokenizer=tokenizer)
    expred_input_wrapper.preprocess()

    expred_output_wrapper = model(expred_input_wrapper)  # The output
    return expred_output_wrapper.get('cls_preds').get('cls_pred')


def lime(lime_tensor_docs, lime_queries, lime_docs, lime_labels, target, n_samples):
    """
    The LIME base is defined here (from the library Captum).

    Args:
        lime_tensor_docs: The docs from the data in tensor form.
        lime_queries: The queries from the data.
        lime_docs: The docs from the data.
        lime_labels: The label from the data.
        target: The index of the highest score of the predictions.
        n_samples: The amount of iterations or samples that the feature attribution method will iterate over.

    Returns:
        A list with the scores for the tokens.
    """
    li = LimeBase(wrapper,
                  interpretable_model=SkLearnLinearRegression(),
                  similarity_func=similarity_kernel,
                  perturb_func=perturb_func,
                  perturb_interpretable_space=False,
                  from_interp_rep_transform=None,
                  to_interp_rep_transform=to_interp_transform)
    attributions = li.attribute(lime_tensor_docs,
                                target=target,
                                additional_forward_args=(lime_queries, lime_docs, lime_labels),
                                n_samples=n_samples,
                                show_progress=True)
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

    with open('/dir/to/your/colorized_lime_base_' + name + '.html', 'w') as f:
        f.write('<p style="word-wrap: break-word; word-break: break-all;">' + colored_string + '</p>')


if __name__ == '__main__':
    # Read from the data, specify the location for the data and the docs
    read_data = read_data('/dir/to/your/{test, train}.jsonl')
    location_docs = '/dir/to/your/docs/'
    # The number of the claim, and the amount of iterations for the feature attribution method
    num = 0  # For when you want to run a specific instance
    # num = int(sys.argv[1])  # For when you want to use run.py
    iterations = 300

    with open("/dir/to/your/lime_base_" + str(num) + ".json", "w") as f:
        # Get the queries, docs and label
        queries = [read_data[num].get('query')]
        docs = read_docs(location_docs, read_data[num].get('docids')[0])
        labels = [read_data[num].get('classification')]
        tensor_docs = to_tensor(docs)

        # In-out-put with the explanation and classification of ExPred
        expred_input, expred_output = in_out_put(queries, docs, labels)

        # LIME Base
        classification = expred_output.get('cls_preds').get('cls_pred').detach().numpy()  # For the index of the target
        attribution = lime(tensor_docs, queries, docs, labels, int(numpy.argmax(classification)), iterations)
        attribution = attribution.squeeze(0)
        attribution = attribution.detach().numpy() / numpy.linalg.norm(attribution.detach().numpy())  # Normalize

        # Visualize
        combine_tokens = [tokenizer.decode(i).replace(' ', '') for i in tensor_docs.detach().numpy()[0]]
        result_tokens, result_scores = remove_hash(combine_tokens, attribution)
        marked_sentence(result_tokens, result_scores, 'normalized_without_hashtags_' + str(num))

        # Write to json
        json.dump({
            "amount_of_iterations": iterations,
            "tokens_list": result_tokens,
            "scores_list": list(map(lambda x: float(x), result_scores))
        }, f)
