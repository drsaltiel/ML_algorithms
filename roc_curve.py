from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split

def plot_roc_curve(model, features, target):
    '''
    plots roc curve
    works for sklearn style models with model().fit method and a .predict_proba attribute
    '''
    target_test, target_predicted_proba = split_predict(model, features, target)

    fpr, tpr, thresholds = roc_curve(target_test, target_predicted_proba[:, 1
                                                                         ])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def split_predict(model, features, target):
    '''
    splits data and predicts on test 
    returns: target, target predicted probability
    '''
    train_feat, test_feat, train_target, test_target = train_test_split(
        features, 
        target, 
        train_size=0.5)
    model_f = model.fit(train_feat, train_target)
    target_predicted_proba = model_f.predict_proba(test_feat)
    return test_target, target_predicted_proba