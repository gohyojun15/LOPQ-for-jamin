from .NQ_utils import Dataset, normalize_passage
import collections

BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])


class JsonQADataset(Dataset):
    def __init__(
            self,
            file: str,
            tensorizer,
            special_token: str = None,
            normalize: bool = False,
    ):
        super().__init__(
            special_token=special_token
        )
        self.tensorizer = tensorizer
        self.file = file
        self.data_files = []
        self.normalize = normalize
        # default initialization
        self.eval = False  # if True, return query_tensor, answers
        self.extract_ctx_embedding = False  # if True, return passage_tensor, passage_id, positive_passages
        self.online_train = False  # if True, query_tensor, passage_tensor, passage_id, positive_passages

    def calc_total_data_len(self):
        if not self.data:
            print("Loading all data.")
            self._load_all_data()
        return len(self.data)

    def load_data(self, start_pos: int = -1, end_pos: int = -1):
        if not self.data:
            self._load_all_data()
        if start_pos >= 0 and end_pos >= 0:
            self.data = self.data[start_pos: end_pos]

    def _load_all_data(self):
        self.data_files = get_dpr_files(self.file)
        print("Data files: %s", self.data_files)
        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        print(f"Total clean data size: {len(self.data)}")

    def __getitem__(self, index):
        json_sample = self.data[index]
        query = self._process_query(json_sample["question"])
        positive_ctxs = json_sample["positive_ctxs"]

        def create_passage(ctx: dict):
            return BiEncoderPassage(
                normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
                ctx["title"],
            )
        positive_passages = create_passage(positive_ctxs[0])
        # Tensorize query and passage
        query_tensor = self.tensorizer.tensorize_query(query)
        passage_tensor = self.tensorizer.tensorize_passage(positive_passages)

        if self.eval:
            answer_list = json_sample["answers"]
            answers = "/".join(answer_list)
            return query_tensor, answers
        elif self.extract_ctx_embedding:
            return passage_tensor, positive_ctxs[0]["passage_id"], positive_passages
        elif self.online_train:
            return query_tensor, passage_tensor, positive_ctxs[0]["passage_id"], positive_passages

        return query_tensor, passage_tensor, positive_ctxs[0]["passage_id"]
