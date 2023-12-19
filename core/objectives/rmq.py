def RmseQuantilieObjective(quantile):
    class _RMO(object):
        def calc_ders_range(self, approxes, targets, weights):
            """
            Computes first and second derivative of the loss function
            with respect to the predicted value for each object.

            Parameters
            ----------
            approxes : indexed container of floats
                Current predictions for each object.

            targets : indexed container of floats
                Target values you provided with the dataset.

            weight : float, optional (default=None)
                Instance weight.

            Returns
            -------
                der1 : list-like object of float
                der2 : list-like object of float

            """
            assert len(approxes) == len(targets)
            if weights is not None:
                assert len(weights) == len(approxes)

            result = []
            for index in range(len(targets)):
                residual = targets[index] - approxes[index]
                mask = residual > 0

                der1 = 2 * (-quantile * residual * mask - (1 - quantile) * residual * ~mask)
                der2 = 2 * (quantile * mask + (1 - quantile) * ~mask)

                if weights is not None:
                    der1 *= weights[index]
                    der2 *= weights[index]

                result.append((der1, der2))
            return result

    return _RMO()


def RmseQuantilieMetric(quantile):
    class _RQM:
        def evaluate(self, approxes, targets, weights):
            """
            Evaluates metric value.

            Parameters
            ----------
            approxes : list of lists of float
                Vectors of approx labels.

            targets : list of lists of float
                Vectors of true labels.

            weights : list of float, optional (default=None)
                Weight for each instance.

            Returns
            -------
                weighted error : float
                total weight : float

            """
            assert len(approxes) == 1
            assert len(targets) == len(approxes[0])

            approx = approxes[0]

            error_sum = 0.0
            weight_sum = 0.0

            for i in range(len(approx)):
                w = 1.0 if weights is None else weights[i]
                weight_sum += w

                error = targets[i] - approx[i]
                squared_error = error**2
                multiplier = 1 if error > 0 else 0
                quantile_error = (
                    multiplier * quantile * squared_error
                    + (1 - multiplier) * (1 - quantile) * squared_error
                )
                error_sum += w * quantile_error

            return error_sum, weight_sum

        def is_max_optimal(self):
            """
            Returns whether great values of metric are better
            """
            return False

        def get_final_error(self, error, weight):
            """
            Returns final value of metric based on error and weight.

            Parameters
            ----------
            error : float
                Sum of errors in all instances.

            weight : float
                Sum of weights of all instances.

            Returns
            -------
            metric value : float

            """
            return (error / (weight + 1e-38)) ** 0.5

    return _RQM()
