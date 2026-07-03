import json
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
nltk.download('punkt')
# 初始化分词器
sentence_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()
smooth_fn = SmoothingFunction().method1

def tokenize_text(text, lang='en'):
    if lang == 'zh':
        # 中文按字符切分，避免分词不一致导致BLEU过低
        return list(text.replace(" ", ""))  # 去掉空格后按字符
    else:
        # 英文先分句，再分词
        sentences = sentence_tokenizer.tokenize(text)
        tokens = []
        for sent in sentences:
            tokens.extend(word_tokenizer.tokenize(sent))
        return tokens

def compute_bleu_from_json(json_file, lang='en'):
    references = []
    hypotheses = []

    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            gt_tokens = tokenize_text(item['Ground truth'], lang)
            pred_tokens = tokenize_text(item['answer'], lang)
            references.append([gt_tokens])  # corpus_bleu要求list of list of refs
            hypotheses.append(pred_tokens)

    # BLEU1~4
    weights_list = [
        (1.0, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (0.33, 0.33, 0.33, 0),
        (0.25, 0.25, 0.25, 0.25)
    ]
    bleu_scores = []
    for i, w in enumerate(weights_list):
        score = corpus_bleu(references, hypotheses, weights=w, smoothing_function=smooth_fn)
        bleu_scores.append(score)
        print(f"{lang.upper()} BLEU-{i+1}: {score:.4f}")

    # 计算 BLEU1、BLEU2、BLEU4 的均值
    mean_score = (bleu_scores[0] + bleu_scores[1] + bleu_scores[3]) / 3
    print(f"{lang.upper()} BLEU1,2,4 平均值: {mean_score:.4f}\n")

if __name__ == "__main__":
    # 示例：依次评估英文和中文文件
    compute_bleu_from_json("outputs/eval/answer_en.json", lang='en')
    compute_bleu_from_json("outputs/eval/answer_zh.json", lang='zh')
