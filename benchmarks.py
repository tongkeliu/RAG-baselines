import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import warnings
warnings.filterwarnings("ignore")
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk

# nltk.download('punkt_tab', quiet=True)   # 新版句子分割器
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)

def compute_meteor(candidate: str, reference: str) -> float:
    """
    计算单条候选文本与参考文本之间的 METEOR 分数。
    
    Args:
        reference (str): 参考文本（真实 review）
        candidate (str): 候选文本（模型生成的 review）
    
    Returns:
        float: METEOR 分数 (0.0 ~ 1.0)
    """
    ref_tokens = word_tokenize(reference.lower())
    cand_tokens = word_tokenize(candidate.lower())
    return meteor_score([ref_tokens], cand_tokens)

def compute_bleu(candidate: str, references: list[str]) -> float:
    """
    计算 sentence-level BLEU-4（带平滑）
    candidate: 生成的句子（字符串）
    references: 参考句子列表（每个是字符串）
    """
    candidate_tokens = candidate.split()
    reference_tokens = [ref.split() for ref in references]
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        smoothing_function=smoothie,
        weights=(0.25, 0.25, 0.25, 0.25)  # BLEU-4
    )
    return bleu


def compute_rouge(candidate: str, reference: str):
    """
    计算 ROUGE-L, ROUGE-1, ROUGE-2
    注意：rouge-score 要求输入为原始字符串（不分词）
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def compute_ppl(text: str, model_name: str = "gpt2") -> float:
    """
    使用 GPT-2 计算文本的 perplexity
    text: 输入文本（字符串）
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    # 添加 EOS token（GPT-2 需要）
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = encodings.input_ids.to(device)
    max_length = model.config.n_positions
    seq_len = input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    with torch.no_grad():
        for begin_loc in range(0, seq_len, max_length):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # target length
            input_ids_window = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_window.clone()
            target_ids[:, :-trg_len] = -100  # 忽略非目标部分

            outputs = model(input_ids_window, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return ppl.item()


if __name__ == "__main__":
    generated = "the cat is on the mat"
    references = ["a cat is sitting on the mat", "there is a cat on the mat"]

    print("Generated:", generated)
    print("References:", references)
    print("-" * 50)

    # BLEU
    bleu = compute_bleu(generated, references)
    print(f"BLEU-4: {bleu:.4f}")

    # ROUGE (通常用第一个参考句，或取平均)
    rouge = compute_rouge(generated, references[0])
    print(f"ROUGE-1: {rouge['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge['rougeL']:.4f}")

    # PPL
    ppl = compute_ppl(generated)
    print(f"Perplexity (PPL): {ppl:.2f}")