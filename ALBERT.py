from transformers import AlbertForQuestionAnswering, AutoTokenizer
from datasets import load_dataset


data = load_dataset("rajpurkar/squad")

# contoh
print("Context: ", data["train"][0]["context"])
print("Question: ", data["train"][0]["question"])
print("Answer: ", data["train"][0]["answers"])

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = AlbertForQuestionAnswering.from_pretrained("albert/albert-base-v2")

print(tokenizer.is_fast)

# test context & question
sample_context = data['train'][0]['context']
sample_question = data['train'][0]['question']

print(sample_context)
print(sample_question)

inputs = tokenizer(sample_question, sample_context)
print(tokenizer.decode(inputs["input_ids"]))

max_length = 384
stride = 128


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs



train_dataset = data["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=data["train"].column_names,
)
len(data["train"]), len(train_dataset)
