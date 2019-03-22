import analyze
import getdata

'''
main method
'''
def main():
    train_alignment,train_reviews = getdata.readFiles("train.npz")
    train_alignment = getdata.decode(train_alignment)
    # train_alignment = getdata.describeLabels(train_alignment)
    train_reviews = getdata.decode(train_reviews)

    test_align,test_reviews = getdata.readFiles("test.npz")
    test_align = getdata.decode(test_align)
    # test_align = getdata.describeLabels(test_align)
    test_reviews = getdata.decode(test_reviews)



    model = analyze.train(train_reviews,test_reviews,train_alignment)
    # analyze.predict(test_reviews,model)
    analyze.predict(test_reviews,test_align,model)
main()
