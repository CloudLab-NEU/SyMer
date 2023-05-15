from nltk.translate.bleu_score import corpus_bleu
import subprocess
import os
from rouge import Rouge


def calculate_bleu(refs, pred):
    Ba = corpus_bleu(refs, pred)
    B1 = corpus_bleu(refs, pred, weights=(1, 0, 0, 0))
    B2 = corpus_bleu(refs, pred, weights=(0, 1, 0, 0))
    B3 = corpus_bleu(refs, pred, weights=(0, 0, 1, 0))
    B4 = corpus_bleu(refs, pred, weights=(0, 0, 0, 1))

    Ba = round(Ba * 100, 2)
    B1 = round(B1 * 100, 2)
    B2 = round(B2 * 100, 2)
    B3 = round(B3 * 100, 2)
    B4 = round(B4 * 100, 2)

    return len(pred), Ba, B1, B2, B3, B4


def split2differFile(filePath):
    refs = list()
    preds = list()
    with open(filePath, "r") as file:
        for i, line in enumerate(file):
            pre, tgt = line.strip('\n').split('|')
            refs.append(tgt)
            preds.append(pre)
    with open(filePath.strip(".txt") + "_refs.txt", 'w') as file:
        file.writelines("\n".join('%s' % r for r in refs))
    with open(filePath.strip(".txt") + "_preds.txt", 'w') as file:
        file.writelines("\n".join('%s' % p for p in preds))


def get_eval_result(predictions, refs):
    true_positive, false_positive, false_negative = 0, 0, 0

    for pre, tgt in zip(predictions, refs):
        for word in pre:
            if word in tgt:
                true_positive += 1
            else:
                false_positive += 1

        for word in tgt:
            if word not in pre:
                false_negative += 1

    return true_positive, false_positive, false_negative


def calculate_meteor(file_name):
    split2differFile("output/{}.txt".format(file_name))
    pred_file = "output/{}_preds.txt".format(file_name)
    ref_file = "output/{}_refs.txt".format(file_name)
    get_meteor = "java -Xmx2G -jar meteor-1.5/meteor-1.5.jar {} {} -l en -norm".format(
        pred_file,
        ref_file)
    p = subprocess.Popen(get_meteor, shell=True, stdout=subprocess.PIPE)
    out, err = p.communicate()
    out = str(out, encoding='utf-8')
    lines = []
    for line in out.splitlines():
        lines.append(line)
    ret = lines[-1]
    ret = round(float(ret.replace("Final score:", "").strip()) * 100, 2)

    os.remove(pred_file)
    os.remove(ref_file)
    return ret


def evaluate(file_name, change_name=False, return_dict=False):
    with open("output/{}.txt".format(file_name), 'r') as file:
        refs = list()
        preds = list()
        for i, line in enumerate(file):
            pre, tgt = line.strip('\n').split('|')
            refs.append(tgt.strip(' .'))
            preds.append(pre.strip(' .'))

    rouge = Rouge()
    rouge_score = rouge.get_scores(preds, refs, ignore_empty=True, avg=True)

    refs = [[r.split(' ')] for r in refs]
    preds = [p.split(' ') for p in preds]

    function_num, Ba, B1, B2, B3, B4 = calculate_bleu(refs, preds)

    meteor_score = calculate_meteor(file_name)
    print("BLEU:  Ba {}  B1 {}  B2 {}  B3 {}  B4 {}".format(
        Ba, B1, B2, B3, B4
    ))

    for key, value in rouge_score.items():
        rouge_score[key] = {k: round(v * 100, 2) for k, v in value.items()}

    print("Rouge: {}".format(rouge_score))

    print("Meteor: {}".format(meteor_score))

    if change_name:
        os.rename("output/{}.txt".format(file_name),
                  "output/{}_Ba{}_Me{}_Rl{}.txt".format(file_name, Ba, meteor_score, rouge_score['rouge-l']['f']))
    if return_dict:
        return {'rouge': rouge_score, 'ba': Ba, 'meteor': meteor_score}
    else:
        return rouge_score['rouge-l']['f'], Ba, meteor_score
