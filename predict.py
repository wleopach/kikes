from hdbscan import prediction
from utils import check_is_df, extract_numerical, extract_categorical, normalize_new
from senseclus import SenseClus


class Pred:

    def __init__(self, clf: SenseClus):
        """
        Each instance of the class Pred must receive a DenseClus object
        :param clf:
        :return: Initializer
        """
        self.clf = clf

    def embed(self, data):
        """
        Embeds new data using the intersection_union_mapper in the DenseClus
        :param data: original data to be labeled with the same shape as the
               data used for the DenseCLus fitting
        :return: embedded data
        """
        check_is_df(data)
        numerical_ = extract_numerical(data)
        categorical_ = extract_categorical(data)
        categorical_umap_emb = self.clf.categorical_umap_.transform(categorical_)
        numerical_umap_emb = self.clf.numerical_umap_.transform(numerical_)
        intersection_umap_emb = self.clf.intersection_umap_.transform(numerical_)

        if self.clf.umap_combine_method == "intersection":
            embed_data_ = normalize_new(numerical_umap_emb *
                                        categorical_umap_emb,
                                        self.clf.min_val, self.clf.max_val)

        elif self.clf.umap_combine_method == "union":
            embed_data_ = normalize_new(numerical_umap_emb +
                                        categorical_umap_emb,
                                        self.clf.min_val, self.clf.max_val)

        elif self.clf.umap_combine_method == "contrast":
            embed_data_ = normalize_new(numerical_umap_emb -
                                        categorical_umap_emb,
                                        self.clf.min_val, self.clf.max_val)

        elif self.clf.umap_combine_method == "intersection_union_mapper":
            embed_data_ = normalize_new(intersection_umap_emb *
                                        (numerical_umap_emb +
                                         categorical_umap_emb),
                                        self.clf.min_val, self.clf.max_val)
        else:
            embed_data_ = None
        return embed_data_

    def predict_new(self, data):
        """
        Predicts labels for unlabeled data
        :param clf: hdbscan pretrained model with predict_data set True
        :param data: embedded data
        :return:predicted labels and probabilities
        """
        pred = prediction.approximate_predict(self.clf.hdbscan_, self.embed(data))
        return pred
