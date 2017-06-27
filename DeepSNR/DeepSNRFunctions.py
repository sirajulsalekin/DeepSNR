import numpy as np


class DeepSNRFunctions(object):

    def OneHotEncoding(self, lines, len_kernel, no_samples):
        A = np.array([[1.0, 0, 0, 0]])
        C = np.array([[0, 1.0, 0, 0]])
        G = np.array([[0, 0, 1.0, 0]])
        T = np.array([[0, 0, 0, 1.0]])

        All_S_Test = 0.25 * np.ones((no_samples, 400 + len_kernel * 8), dtype=np.float32)
        All_S_Test[:, len_kernel * 4:len_kernel * 4 + 400] = np.zeros((no_samples, 400), dtype=np.float32)

        for k in range(0, no_samples):
            in_seq = lines[k]

            for j in range(0, 100):
                if in_seq[j] == 'A':
                    All_S_Test[k, (len_kernel + j) * 4:(len_kernel + j + 1) * 4] = A
                elif in_seq[j] == 'C':
                    All_S_Test[k, (len_kernel + j) * 4:(len_kernel + j + 1) * 4] = C
                elif in_seq[j] == 'G':
                    All_S_Test[k, (len_kernel + j) * 4:(len_kernel + j + 1) * 4] = G
                else:
                    All_S_Test[k, (len_kernel + j) * 4:(len_kernel + j + 1) * 4] = T

        return All_S_Test

    def PerformanceEval(self, y_out, test_label, no_samples):
        eval_metric = np.zeros((no_samples, 4), dtype=np.float)

        for i in range(no_samples):
            Int_DSNR = float(sum(np.all([y_out[i, :], test_label[i, :] == 1], axis=0)))
            Uni_DSNR = float(sum(np.any([y_out[i, :], test_label[i, :] == 1], axis=0)))
            eval_metric[i, 0] = Int_DSNR/sum(y_out[i, :])  # precision
            eval_metric[i, 1] = Int_DSNR/sum(test_label[i, :])  # recall
            eval_metric[i, 2] = 2*eval_metric[i, 0]*eval_metric[i, 1]/(eval_metric[i, 0]+eval_metric[i, 1])  # F1-score
            eval_metric[i, 3] = Int_DSNR/Uni_DSNR  # IoU

        return eval_metric
