
from app.datasets.loader import DatasetLoader


def main():
    loader = DatasetLoader()

    samples = loader.load_jsonl(
        "data/datasets/fire_qa.jsonl"
    )

    print("\n========== 数据集加载测试 ==========")

    print("样本总数：", len(samples))

    verified_count = sum(
        1 for sample in samples if sample.verified
    )

    print("已核验样本：", verified_count)
    print("未核验样本：", len(samples) - verified_count)

    print("\n第一条样本：")

    sample = samples[0]

    print("ID：", sample.sample_id)
    print("问题：", sample.question)
    print("标准答案：", sample.ground_truth)
    print("类别：", sample.category)
    print("难度：", sample.difficulty)
    print("是否核验：", sample.verified)

    print("====================================")


if __name__ == "__main__":
    main()
