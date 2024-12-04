""" 
PRISM data augmenation (reformatting, context augmentation, embeddings, etc.) 
for input into VPL training with cached LLM embeddings. 
"""

import pandas as pd 
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
import os
from typing import List, Dict, Union, Tuple
from dotenv import load_dotenv
import argparse
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
PRISM_DATASET = 'MichaelR207/prism_personalized_1023'



def generate_prism_vpl_data(dataset: object, 
                            data_dir: str = os.path.join(os.getcwd(), 'data'), 
                            dir_name: str = 'prism', 
                            model_type: str = 'gpt2', 
                            context_sample_strategy: str = 'random', 
                            num_random_contexts: int = 5,
                            with_embeddings: bool = False, 
                            embedding_pool_strategy: str = 'last',
                            data_subset: str = 'all', 
                            data_split: str = 'all'    
                            )-> None: 
    """ 
    Michael's preprocessed PRISM data -> VPL data format
    
    Args:
    dataset: object
        The huggingface dataset object. 
    dir_name: str
        The name of the directory where the data will be saved under data/ 
    model_type: str
        The name of the LLM encoder model type to use. Pair and Sequential Encoder
        architectures are fixed. 
    context_sample_strategy: str
        The method for selecting preference pairs to include in the context.
        Options are ['random', 'top2', 'bestworst', 'max', 'cum']
        'top2' only includes the top 2 ranked responses from each turn (based on user score)
        'random' randomly selects comparisons from prior turns and other conversations
        'bestworst' only includes the comparison of the highest and lowest score from each turn
        'max' not only includes everything in the turn, but ALL comparisons at ALL turns in all other conversations
        'cum' will include ALL 5 Choose 2 preference pairs from the PRISM data
        at each prior turn in this same conversation (originating from 5 ranked responses).
    num_random_contexts: int
        The number of random comparisons to include in the context IF context_sample_strategy = 'random'.
        Otherwise, this parameter is ignored.
    with_embeddings: bool
        Whether to pre-generate the embeddings for the various prompt, response pairs 
        This will produce embeddings from the `model_type` encoder for each chosen pair 
        and rejected pair, both for the current example, and for the contexts. 
        The contexts will not retain original strings when using `with_embeddings`. 
        Highly recommended to cache these for model training, only pass False when 
        you want to manually inspect the data.
    embedding_pool_strategy: str
        The pooling method to apply to the token embeddings. Options are 'mean' or 'last'.
    data_subset: str
        Which subset (or "type") of data to include. Can be useful for analysis. For PRISM, 
        options are 'all', to include all data or one of:
        ['unguided', 'values guided', 'controversy guided']
    data_split: str
        Which split of the data to generate/ process. For PRISM, 
        options are 'all', to include all data or one of:
        ['train', 'validation', 'test']
    """
    train_df = pd.DataFrame(dataset['train'])
    validation_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])
    dfs = []
    if data_split == 'train':
        dfs = [train_df]
    elif data_split == 'validation':
        dfs = [validation_df]
    elif data_split == 'test':
        dfs = [test_df]
    elif data_split == 'all':
        dfs = [train_df, validation_df, test_df]
    for df in dfs:
        """ 
        4. Map the prompt only to `prompt` field. 
        5. Map the two responses in (shown) order to `responses` field, a List[str]. If flip is true, show the rejected response first.
        1. Append prompt and response together with Human: {prompt}\n\nAssistant: {response}. Do this to create both `chosen` and `rejected` fields for the current turn (the preference pair we are trying to predict). 
        2. Generate an 'Index' field. This should just be unique to the current row. 
        3. Make a dummy `data_subset` using "type" (e.g. unguided). In the VPL work they use the data_subset for various visualizations. If we later do some analysis to cluster various types of prompt-response challenges in the PRISM data this may be useful. 
        6. If the first response in the `responses` field was the preferred response, set `original_label`: 0 and `label`: 0. Otherwise set both to 1. 
        7. Map `objective` to the same thing as `data_subset`. 
        8. Set `controversial` to True only for controversy guided. I believe in the original paper this was marked for further analysis to emphasize that VPL works better on examples asking controversial/split responses, e.g. comparing cats vs dogs to be better in the pets dataset. The exact definition is left unclear in the OG paper. 
        11. Create a NEW `contexts` field. Each item in the `contexts` field will contain the keys `embedding_chosen`, `embedding_rejected`, and `original_id` (I think this `original_id` is just the row ID of the example used to create `embedding_chosen` and `embedding_rejected`). This is the field to be most careful in creating. + there are some hyperparameters to experiment with here, down the line. For all prior turns in the conversation (1 prior turn minimum, we will opt to NOT predict the first turn, based on other preferences for the same turn as that feels leaky to me (/IS leaky)), we can use ALL 5 choose 2 preference pairs in the dataset and add those to the NEW `contexts` list, or we can just choose the favorite vs. least favorite response (or fav vs. 2nd most fav response) from each prior turn. We can also sample prior turns when there have been many turns to minimize the length and 'soupiness' of the contexts. All of these will affect the learning of the Pair Encoder and Sequential Encoder in the VPL architecture and are thus important choices. So long as we only use embedding pairs from previous (chronologically) turns for the given conversation and user (or from any point in another conversation with this user), then we are not leaking data into the contexts. 
        9. Use a similar framework as in `vpl_llm/vpl_modules/data_utils/data_processing.generate_embeddings_with_llm()` to generate embeddings for the `chosen` and `rejected` ($s_A$ and $s_B$, respectively, with $[p,r]$ (prompts and responses concatenated). This is essential mapping the `chosen` and `rejected` fields from (1) to a hidden encoded state from the chosen pre-trained LLM (such as GPT2/ llama3). Map these to `embeddings` as a Dict[str, List[float]], where in the dictionary we have keys `embedding_chosen` with value = the LLM embedding of `chosen`, and the same for `embedding_rejected`.)
        10. Create a `context_length` which is the length of the NEW FIELD `contexts`. This will NOT match the length of the original `contexts` field in Michael's data, as that is a list of each and every turn (prompt, response, prompt, response, prompt) etc., always ending with the current turn's prompt. Rather, the new contexts field will contain embeddings for previous `embedding_chosen` and `embedding_rejected` for the current user. In the (prompt, response, prompt, response, prompt) example, this means we should expect the NEW `context_length` to be 2, since we will include the two prior $[p,r]$ turn encoding dictionaries. This length will be multiplied if we include multiple comparison (e.g. all 5C2 comparisons) for each turn. For more details, see (11). 
        """
        CONTROVERSY_TYPE = 'controversy guided'   #Hf 'type' for controversy guided conversations
        
        df['prompt'] = df['context'].map(lambda x: x[-1]['content'])
        def get_displayed_response_list(x):
            if not bool(x['flip']):
                return [x['chosen']['content'], x['rejected']['content']]
            else:
                return [x['rejected']['content'], x['chosen']['content']]
        df['responses'] = df.loc[:, ['chosen', 'rejected', 'flip']].apply(
                lambda x: get_displayed_response_list(x), axis=1
            )
        def get_label(x):   # label indicates the index of the preferred response in the responses list
            return int(x)  # if flip is True, label 1 else 0 
        df['original_label'] = df['flip'].map(lambda x: get_label(x))
        df['label'] = df['original_label']
        df['chosen'] = 'Human: ' + df['prompt'] + '\n\nAssistant: ' + df['chosen'].map(lambda x: x['content'])
        df['rejected'] = 'Human: ' + df['prompt'] + '\n\nAssistant: ' + df['rejected'].map(lambda x: x['content'])
        df['Index'] = df.index
        df['data_subset'] = df['type']
        df['objective'] = df['type']
        df['controversial'] = df['type'].map(lambda x: x == CONTROVERSY_TYPE)
        
        if with_embeddings: 
            df = get_llm_embeddings(df, 
                                    model_type = model_type)
        
        # ===== augment the context field with chosen and rejected pairs from prior turns =====
        df = augment_contexts(df,
                    with_embeddings = with_embeddings, 
                    which_comparisons = context_sample_strategy,
                    num_random = num_random_contexts,
                    seed=329)
        # ======================================================================================
        df['contexts'] = df['augmented_context']
        df['context_length'] = df['contexts'].map(lambda x: len(x))
        
        print(df.columns)
        df = clean_df(df)

        if data_subset != 'all':
            df = df[df['type'] == data_subset]
            
            
    output_dir = os.path.join(data_dir, dir_name, model_type, f"context_{context_sample_strategy}", f"numrandom_{num_random_contexts}", f"pooling_{embedding_pool_strategy}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if data_split == 'train':
        train_df.to_json(os.path.join(output_dir, 'train.jsonl'), orient="records", lines=True, index=False)
    elif data_split == 'validation':
        validation_df.to_json(os.path.join(output_dir, 'validation.jsonl'), orient="records", lines=True, index=False)
    elif data_split == 'test':
        test_df.to_json(os.path.join(output_dir, 'test.jsonl'), orient="records", lines=True, index=False)
    elif data_split == 'all':
        train_df.to_json(os.path.join(output_dir, 'train.jsonl'), orient="records", lines=True, index=False)
        validation_df.to_json(os.path.join(output_dir, 'validation.jsonl'), orient="records", lines=True, index=False)
        test_df.to_json(os.path.join(output_dir, 'test.jsonl'), orient="records", lines=True, index=False)
        
    return None
    
            
def augment_contexts(df: pd.DataFrame,
            with_embeddings: bool = False, 
            which_comparisons: str = 'random',
            num_random: int = 5, 
            seed: int = 329):
    """
    Input: 
    df: pd.DataFrame
        The dataframe containing the PRISM data in the VPL format. 
    which_comparisons: str
        The method for selecting preference pairs to include in the context. 
        TODO: add other options (e.g. 'top2', 'random', 'best_worst', 'max', 'all')
        'top2' only includes the top 2 ranked responses from each turn (based on user score) 
        'random' randomly selects comparisons from prior turns and other conversations
        'bestworst' only includes the comparison of the highest and lowest score from each turn 
        'max' not only includes everything in the turn, but ALL comparisons at ALL turns in all other conversations
        'cum' will include ALL 5 Choose 2 preference pairs from the PRISM data 
        at each prior turn in this same conversation (originating from 5 ranked responses). 
    num_random: int
        The number of random comparisons to include in the context IF which_comparisons = 'random'.capitalize.
        Otherwise, this parameter is ignored.
    Output: 
    df: pd.DataFrame
        The dataframe with the `context` field augmented with the chosen and rejected pairs 
        from the prior turns of the conversation. Also contains a field original_id which 
        contains the row index of the example in the data used to grab these chosen and rejected 
        concatenations.
    """
    # Details: 
        # maintain a dictionary which maps user_id, conversation_id, turn to a list of dictionaries
        # each dictionary will contain 'chosen', 'rejected', and 'original_id' fields. 
        # we extend the list for each (user_id, conversation_id, turn) index in the dictionary when encountering 
        # a row with this index. We add a new dictionary containing the chosen 
        # and rejected strings from this row and 'orinal_id' as the index of this row. 
        
        # iterate back through dataframe and create 'augmented_context' field 
        # which contains a list of chosen,rejected,original_id dictionaries. 
        # This list is generated in different ways depending on the which_comparisons parameter.
        # It can be all comparisons from all prior turns, only the top 2, only the best and worst, a random 
        # sample from this/other conversations etc. 
        # IMPORTANT: ALWAYS the comparisons cannot be from the current turn or later in the same 
        # conversation.
        
        # return the dataframe with the 'augmented_context' field added.
    CHOSEN_FIELD = 'embedding_chosen' if with_embeddings else 'chosen'
    REJECTED_FIELD = 'embedding_rejected' if with_embeddings else 'rejected'
    np.random.seed(seed)

    user_convo_turns = df[['user_id', 'conversation_id', 'turns']].drop_duplicates()
    user_convo_turns = list(user_convo_turns.itertuples(index=False, name=None))
    user_convo_turn_dict = {k: [] for k in user_convo_turns}
    
    # ==== fill in the user_convo_turn_dict ====
    for row in df.itertuples():
        user_convo_turn = (row.user_id, row.conversation_id, row.turns)
        if user_convo_turn not in user_convo_turn_dict:
            raise ValueError(f"User conversation turn tuple {user_convo_turn} not found in user_convo_turn_dict.")
        
        user_convo_turn_dict[user_convo_turn].append({
            CHOSEN_FIELD: getattr(row, CHOSEN_FIELD),
            REJECTED_FIELD: getattr(row, REJECTED_FIELD),
            'original_id': row.Index
        })

    augmented_contexts = [None] * len(df)
    # ==== iterate back through the dataframe and create the augmented_context field ====
    if which_comparisons == 'random':
        for i, row in enumerate(df.itertuples()):
            user_id = row.user_id
            conversation_id = row.conversation_id
            turn = row.turns
            user_convo_turn = (user_id, conversation_id, turn)
            augmented_context = []
            for _ in range(num_random):
                random_user_convo_turn = user_convo_turn
                random_turn = turn 
                random_conversation_id = conversation_id
                while random_conversation_id == conversation_id and random_turn >= turn:   # Don't include current turn or later in the same conversation as context
                    random_user_convo_turn = user_convo_turns[np.random.choice(len(user_convo_turns))]
                    random_conversation_id = random_user_convo_turn[1]
                    random_turn = random_user_convo_turn[2]
                chosen_comparison_list = user_convo_turn_dict[random_user_convo_turn]
                random_chosen_rejected = np.random.choice(chosen_comparison_list)   # also randomly select the chosen and rejected pair.
                augmented_context.append(random_chosen_rejected)
            augmented_contexts[i] = augmented_context
    else:
        raise NotImplementedError(f"which_comparisons = {which_comparisons} not yet implemented.")
    
    df['augmented_context'] = augmented_contexts
    
    return df

def get_llm_embeddings(df: pd.DataFrame, 
                    model_type: str, 
                    embed_dim: int = 768, 
                    pooling: str = 'last') -> pd.DataFrame:
    """
    Generates embeddings for 'chosen' and 'rejected' fields in the dataframe using a specified language model.

    Args:
        df (pd.DataFrame): The DataFrame containing 'chosen' and 'rejected' fields.
        model_type (str): The model type to use for generating embeddings ('gpt2', 'llama').
        embed_dim (int): Dimension of the embeddings (default is 768 for many models).
        pooling (str): Pooling method to apply to the token embeddings. Options are 'mean' or 'last'.

    Returns:
        pd.DataFrame: The original DataFrame with added 'embedding_chosen' and 'embedding_rejected' columns.
    """
    
    # Load model and tokenizer based on model type
    if model_type == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2", token = HUGGINGFACE_TOKEN)
        model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=embed_dim, torch_dtype=torch.bfloat16)
        model.score.weight.data *= 0.01
    elif model_type == "meta-llama/Llama-2-7b-hf":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token = HUGGINGFACE_TOKEN, add_eos_token=False)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
    elif model_type == "meta-llama/Llama-3.1-8B-Instruct":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", token = HUGGINGFACE_TOKEN, add_eos_token=False)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")
    
    model.to("cuda")  # Move model to GPU for faster processing
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id

    embeddings_chosen = []
    embeddings_rejected = []

    # Generate embeddings for each row
    for row in tqdm(df.itertuples(), total=len(df)):
        row_embeddings = {}

        for key in ['chosen', 'rejected']:
            text = getattr(row, key)
            tokens = tokenizer.pad(
                tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512),
                padding=True, pad_to_multiple_of=64
            )
            input_ids = tokens["input_ids"].to("cuda")
            attention_mask = tokens["attention_mask"].to("cuda")

            with torch.no_grad():
                last_hidden_state = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                ).hidden_states[-1]
                
                # Choose pooling method
                if pooling == 'mean':
                    # Take the mean of the last hidden state for the embedding
                    embedding = last_hidden_state.mean(dim=1).squeeze().to(torch.float32).cpu().numpy()
                elif pooling == 'last':
                    # Take the last token's hidden state as the embedding
                    token_length = len(tokens["input_ids"][0].nonzero(as_tuple=True)[0])  # count non-padding tokens
                    embedding = last_hidden_state[0][token_length - 1].float().cpu().numpy()
                else:
                    raise ValueError(f"Pooling method '{pooling}' not recognized. Choose 'mean' or 'last'.")

            # Store embeddings based on the key
            row_embeddings[f"embedding_{key}"] = embedding

        embeddings_chosen.append(row_embeddings["embedding_chosen"])
        embeddings_rejected.append(row_embeddings["embedding_rejected"])

    # Add the embeddings to the DataFrame
    df['embedding_chosen'] = embeddings_chosen
    df['embedding_rejected'] = embeddings_rejected
    
    # stuff them into same dictionary to be consistent with VPL method
    df['embeddings'] = df.apply(
        lambda x: {'embedding_chosen': x['embedding_chosen'], 'embedding_rejected': x['embedding_rejected']},
        axis=1
    )

    return df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame by retaining only necessary columns and structuring it 
    to match the desired JSON output format for VPL.

    Args:
        df (pd.DataFrame): The original DataFrame with potentially extra columns.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for export in the desired JSON format.
    """
    # Keep only the data needed for VPL
    required_columns = [
        'chosen', 'rejected', 'Index', 'data_subset', 'prompt', 'responses',
        'original_label', 'objective', 'label', 'controversial', 'embeddings', 
        'context_length', 'contexts'
    ]
    df = df[required_columns]
    df = df.rename(columns={'Index': 'index'})

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate PRISM data for VPL training with cached LLM embeddings")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory name")
    parser.add_argument("--model_type", type=str, default="gpt2", choices=["gpt2", "llama"], help="Model type for generating embeddings")
    parser.add_argument("--context_sample_strategy", type=str, default="random", choices=["random", "top2", "bestworst", "max", "cum"], help="Strategy for sampling context pairs")
    parser.add_argument("--num_random_contexts", type=int, default=5, help="Number of random context pairs to sample")
    parser.add_argument("--with_embeddings", action="store_true", help="Flag to generate embeddings and cache them ahead of time for faster VPL training (and inference).")
    parser.add_argument("--embedding_pool_strategy", type=str, default="last", choices=["mean", "last"], help="Pooling strategy for generating embeddings for input strings from the LLM. 'last' recommended for autoregressive models.")
    parser.add_argument("--data_subset", type=str, default="all", help="Subset of data (e.g., 'all', 'unguided', etc.)")
    parser.add_argument("--data_split", type=str, required=True, help="Data split (e.g., 'train', 'validation', 'test')")

    args = parser.parse_args()

    dataset = load_dataset(PRISM_DATASET)
    generate_prism_vpl_data(dataset=dataset, 
                            dir_name=args.output_dir, 
                            model_type=args.model_type, 
                            context_sample_strategy=args.context_sample_strategy,
                            num_random_contexts=args.num_random_contexts,
                            with_embeddings=args.with_embeddings, 
                            embedding_pool_strategy=args.embedding_pool_strategy,
                            data_subset=args.data_subset, 
                            data_split=args.data_split)

if __name__ == "__main__":
    main()