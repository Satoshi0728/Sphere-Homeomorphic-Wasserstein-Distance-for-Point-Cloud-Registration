import pandas as pd

"""
点群データの保存
ソース点群と、ターゲット点群、変換後のソース点群を保存する
"""
class Save_point_cloud():
    def __init__(self,point_cloud_save_dir):
        super().__init__()
        self.point_cloud_save_dir = point_cloud_save_dir

    def __call__(self, source, target, transformed_source, epoch):
        self.source = source
        self.target = target
        self.transformed_source = transformed_source
        #ソース点群の保存
        for i in range(1):
            source = self.source[i]
            source = source.detach().cpu().numpy()
            pd.to_pickle(source, self.point_cloud_save_dir + '/' + 'initial_source_data/' + f"initial_data_{epoch:06}.pkl")
        #ターゲット点群の保存
        for i in range(1):
            target = self.target[i]
            target = target.detach().cpu().numpy()
            pd.to_pickle(target, self.point_cloud_save_dir + '/' + 'target_data/' + f"target_data_{epoch:06}.pkl")
        #変換後のソース点群の保存
        for i in range(1):
            transformed_source = self.transformed_source[i]
            transformed_source = transformed_source.detach().cpu().numpy()
            pd.to_pickle(transformed_source, self.point_cloud_save_dir + '/' + 'transformed_source_data/' + f"transformed_data_{epoch:06}.pkl")



