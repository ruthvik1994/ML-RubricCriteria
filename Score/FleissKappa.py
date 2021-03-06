import numpy as np

class FleissKappa(object):

    def __init__(self, artifacts):
        self.data = artifacts
        self.mat = []
        max_category = -1
        for artifact_id in artifacts:
            row = [int(i) for i in artifacts[artifact_id]]
            max_category = max(max_category, max(row))
        for artifact_id in artifacts:
            row = [int(i) for i in artifacts[artifact_id]]
            try:
                if len(row) < 6:
                    sample = np.random.choice(row, size=6, replace=True)
                else:
                    sample = np.random.choice(row, size=6, replace=False)
                k = [0 for _ in range(1, max_category + 1)]
                for i in sample:
                    k[i-1] += 1
                self.mat.append(k)
            except:
                print("Error")

    @staticmethod
    def compute(mat):
        num_raters = sum(mat[0])
        num_artifacts = len(mat)
        num_categories = len(mat[0])
        # Computing p[]
        p = [0.0] * num_categories
        for j in range(num_categories):
            p[j] = 0.0
            for i in range(num_artifacts):
                p[j] += mat[i][j]
            p[j] /= num_artifacts * num_raters

        # Computing P[]
        P = [0.0] * num_artifacts
        for i in range(num_artifacts):
            P[i] = 0.0
            for j in range(num_categories):
                P[i] += mat[i][j] * mat[i][j]
            P[i] = (P[i] - num_raters) / (num_raters * (num_raters - 1))

        # Computing Pbar
        Pbar = sum(P) / num_artifacts
        # Computing PbarE
        PbarE = 0.0
        for pj in p:
            PbarE += pj * pj
        try:
            kappa = (Pbar - PbarE) / (1 - PbarE)
            return kappa
        except:
            print("Error: Divide by zero")
            return None
