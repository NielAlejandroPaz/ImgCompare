#ifndef IMGCOMPARE_H
#define IMGCOMPARE_H

#include <QMainWindow>
#include <QFileDialog>
#include <unordered_map>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <unordered_set>
#include <iostream>

QT_BEGIN_NAMESPACE
namespace Ui { class ImgCompare; }
QT_END_NAMESPACE

class ImgCompare : public QMainWindow
{
    Q_OBJECT

    struct structImg {
        QString imgPath;
        cv::Mat hash;
        std::vector<QString> duplicates;
        structImg(const QString& p, const cv::Mat& m) : imgPath(p), hash(m) {};
    };

public:
    ImgCompare(QWidget *parent = nullptr);
    ~ImgCompare();

signals:
    void setProgressBarValue(int val);

private slots:
    void folderBrowseClicked();
    void analyseClicked();
    void similarityValueChanged(int val);
    void tableSelectionChanged(int row, int col);
    void duplicateForwardClicked();
    void duplicateBackwardClicked();
    void fillWidgetTable(bool onlyDuplicates);

private:
    Ui::ImgCompare *ui;
    std::vector<QString> getImagesFromDirectory(std::string rootpath);
    void updateDuplicateImage();
    void deinitGUIComponents();

    // Based on opencv_contrib img_hash by @vrabaud
    // https://github.com/opencv/opencv_contrib/tree/master/modules/img_hash
    //-------------
    void getMoments(double *pointer, const std::vector<cv::Mat>& channels);
    void getHash(cv::InputArray inputArr, cv::OutputArray outputArr);
    double compareHashes(cv::InputArray hash1, cv::InputArray hash2);
    //-------------

    // other image comparision functions
    void getHistogram(cv::InputArray img, cv::OutputArray histogram);
    double getPSNR(const cv::Mat& I1, const cv::Mat& I2);
    cv::Scalar getMSSIM( const cv::Mat& i1, const cv::Mat& i2);

    std::vector<structImg> hashedImages_;
    int currentlySelectedDuplicateIdx_;
    int similarityValue_;
    std::unordered_map<int, int> tableToHash_;
};
#endif // IMGCOMPARE_H
