#include "imgcompare.h"
#include "./ui_imgcompare.h"

ImgCompare::ImgCompare(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::ImgCompare)
{
    ui->setupUi(this);
    QObject::connect(this, SIGNAL(setProgressBarValue(int)), this->ui->analyse_progressBar, SLOT(setValue(int)));
    similarityValue_ = 90;
}

ImgCompare::~ImgCompare()
{
    delete ui;
}

void ImgCompare::folderBrowseClicked()
{
    QFileDialog dialog(this);
    dialog.setFileMode(QFileDialog::Directory);
    ui->folder_lineEdit->setText(dialog.getExistingDirectory());
}

void ImgCompare::analyseClicked()
{
    QString rootFolder = ui->folder_lineEdit->text();
    if (rootFolder == nullptr)
    {
         ui->statusbar->showMessage("No folder loaded!", 2000);
        return;
    }

    //de-init
    hashedImages_.clear();
    ui->onlyDuplicates_checkBox->setChecked(false);
    std::vector<QString> imageList = getImagesFromDirectory(rootFolder.toStdString());

    int totalImages = imageList.size();
    int counter = 0;
    double similarityIdx = 100 - similarityValue_;

    for(const auto& path : imageList)
    {
        cv::Mat img = cv::imread(path.toStdString(),cv::IMREAD_ANYCOLOR);
        cv::Mat hash;
        getHash(img, hash);
        bool duplicated = false;
        for (auto& hashedImg : hashedImages_)
        {
            if(compareHashes(hash, hashedImg.hash) <= similarityIdx )
            {
                hashedImg.duplicates.emplace_back(path);
                duplicated = true;
            }
        }
        if (!duplicated)
        {
            structImg newImg(path, hash);
            hashedImages_.emplace_back(newImg);
        }

        emit setProgressBarValue(++counter * 100 / totalImages);
    }

    //add the struct array to tablewidget
    ui->toolBox->setCurrentIndex(1);
    fillWidgetTable(false);
}

void ImgCompare::similarityValueChanged(int val)
{
    ui->similarityIdx_label->setText( QString::number(val) + "%");
    similarityValue_ = val;
}

void ImgCompare::tableSelectionChanged(int row, int col)
{
    int idx = tableToHash_[row];
    QPixmap img(hashedImages_[idx].imgPath);
    int w = ui->img1->width();
    int h = ui->img1->height();
    ui->img1->setPixmap(img.scaled(w, h, Qt::KeepAspectRatio, Qt::SmoothTransformation));

    currentlySelectedDuplicateIdx_ = 0;
    int duplicateSize = hashedImages_[idx].duplicates.size();
    if(duplicateSize > 0)
    {
        ui->forward_pushButton->setEnabled(true);
        ui->backward_pushButton->setEnabled(true);
        duplicateForwardClicked();
    }
    else
    {
        ui->img2->setText("No duplicates for this image");
        ui->duplicate_counter_label->setText("0/0");
        ui->forward_pushButton->setEnabled(false);
        ui->backward_pushButton->setEnabled(false);
    }
}

void ImgCompare::duplicateForwardClicked()
{
    int currentRow = tableToHash_[ui->img_tableWidget->currentRow()];
    currentlySelectedDuplicateIdx_++;
    if(currentlySelectedDuplicateIdx_ >= hashedImages_[currentRow].duplicates.size())
    {
        currentlySelectedDuplicateIdx_ = 0;
    }
    updateDuplicateImage();
}

void ImgCompare::duplicateBackwardClicked()
{
    currentlySelectedDuplicateIdx_--;
    if(currentlySelectedDuplicateIdx_ < 0)
    {
        int currentRow = tableToHash_[ui->img_tableWidget->currentRow()];
        currentlySelectedDuplicateIdx_ = hashedImages_[currentRow].duplicates.size() - 1;
    }
    updateDuplicateImage();
}

void ImgCompare::fillWidgetTable(bool onlyDuplicates)
{
    deinitGUIComponents();
    ui->img_tableWidget->setColumnCount(2);

    for (int i = 0, tableIdx = 0; i < hashedImages_.size(); i++)
    {
        int duplicateNumber = hashedImages_[i].duplicates.size();
        if(duplicateNumber == 0 && onlyDuplicates)
        {
            continue;
        }
        ui->img_tableWidget->insertRow(tableIdx);
        QTableWidgetItem *dupNumber = new QTableWidgetItem(QString::number(duplicateNumber));
        ui->img_tableWidget->setItem(tableIdx, 0, dupNumber);

        QTableWidgetItem *newOtherItem = new QTableWidgetItem(hashedImages_[i].imgPath);
        ui->img_tableWidget->setItem(tableIdx, 1, newOtherItem);
        tableToHash_[tableIdx] = i;
        tableIdx++;
    }

    ui->img_tableWidget->setCurrentCell(0,0);
    tableSelectionChanged(0,0);
}

void ImgCompare::updateDuplicateImage()
{
    int currentRow = tableToHash_[ui->img_tableWidget->currentRow()];
    int maxSize = hashedImages_[currentRow].duplicates.size();
    QPixmap img(hashedImages_[currentRow].duplicates[currentlySelectedDuplicateIdx_]);
    int w = ui->img2->width();
    int h = ui->img2->height();
    ui->img2->setPixmap(img.scaled(w,h, Qt::KeepAspectRatio, Qt::SmoothTransformation));

    QString displayDup = QString::number(currentlySelectedDuplicateIdx_+1) + "/" + QString::number(maxSize);
    ui->duplicate_counter_label->setText(displayDup);
}

void ImgCompare::deinitGUIComponents()
{
    currentlySelectedDuplicateIdx_ = 0;
    ui->img1->setText("img1");
    ui->img2->setText("img2");

    ui->img_tableWidget->clearContents();
    ui->img_tableWidget->clear();
    ui->img_tableWidget->setRowCount(0);
    tableToHash_.clear();
}

std::vector<QString> ImgCompare::getImagesFromDirectory(std::string rootpath)
{
    std::vector<QString> imageList;
    for (const auto &file : std::filesystem::directory_iterator(rootpath))
    {
        std::string filepath = file.path();
        if (filepath.find(".png") != std::string::npos || filepath.find(".jpg") != std::string::npos)
            imageList.emplace_back(QString::fromStdString(filepath));
    }

    return imageList;
}

double ImgCompare::getPSNR(const cv::Mat& I1, const cv::Mat& I2)
{
    cv::Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2
    cv::Scalar s = sum(s1);         // sum elements per channel
    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse =sse /(double)(I1.channels() * I1.total());
        double psnr = 10.0*log10((255*255)/mse);
        return psnr;
    }
}

cv::Scalar ImgCompare::getMSSIM( const cv::Mat& i1, const cv::Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;
    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    cv::Mat I2_2   = I2.mul(I2);        // I2^2
    cv::Mat I1_2   = I1.mul(I1);        // I1^2
    cv::Mat I1_I2  = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
    cv::Mat mu1_2   =   mu1.mul(mu1);
    cv::Mat mu2_2   =   mu2.mul(mu2);
    cv::Mat mu1_mu2 =   mu1.mul(mu2);
    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    cv::Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
    cv::Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
    return mssim;
}

void ImgCompare::getMoments(double *inout, const std::vector<cv::Mat>& channels)
{
    for(size_t i = 0; i != channels.size(); ++i)
    {
        cv::HuMoments(cv::moments(channels[i]), inout);
        inout += 7;
    }
}

void ImgCompare::getHash(cv::InputArray inputArr, cv::OutputArray outputArr)
{
    cv::Mat const input = inputArr.getMat();
    cv::Mat colorImg_, resizeImg_, blurImg_, colorSpace_;
    std::vector<cv::Mat> channels_;
    CV_Assert(input.type() == CV_8UC4 ||
            input.type() == CV_8UC3 ||
            input.type() == CV_8U);

    if(input.type() == CV_8UC3)
    {
      colorImg_ = input;
    }
    else if(input.type() == CV_8UC4)
    {
      cv::cvtColor(input, colorImg_, cv::COLOR_BGRA2BGR);
    }
    else
    {
      cv::cvtColor(input, colorImg_, cv::COLOR_GRAY2BGR);
    }

    cv::resize(colorImg_, resizeImg_, cv::Size(512,512), 0, 0, cv::INTER_CUBIC);
    cv::GaussianBlur(resizeImg_, blurImg_, cv::Size(3,3), 0, 0);

    cv::cvtColor(blurImg_, colorSpace_, cv::COLOR_BGR2HSV);
    cv::split(colorSpace_, channels_);
    outputArr.create(1, 42, CV_64F);
    cv::Mat hash = outputArr.getMat();
    hash.setTo(0);
    getMoments(hash.ptr<double>(0), channels_);

    cv::cvtColor(blurImg_, colorSpace_, cv::COLOR_BGR2YCrCb);
    cv::split(colorSpace_, channels_);
    getMoments(hash.ptr<double>(0) + 21, channels_);
}

double ImgCompare::compareHashes(cv::InputArray hash1, cv::InputArray hash2)
{
    return norm(hash1, hash2, cv::NORM_L2) * 10000;
}

void ImgCompare::getHistogram(cv::InputArray img, cv::OutputArray histogram)
{
    //! [Convert to HSV]
    cv::Mat img_hsv;
    cvtColor( img, img_hsv, cv::COLOR_BGR2HSV );

   // [Using 50 bins for hue and 60 for saturation]
   int h_bins = 50, s_bins = 60;
   int histSize[] = { h_bins, s_bins };

   // hue varies from 0 to 179, saturation from 0 to 255
   float h_ranges[] = { 0, 180 };
   float s_ranges[] = { 0, 256 };

   const float* ranges[] = { h_ranges, s_ranges };

   // Use the 0-th and 1-st channels
   int channels[] = { 0, 1 };
   cv::Mat hist_base;

   cv::calcHist( &img_hsv, 1, channels, cv::Mat(), hist_base, 2, histSize, ranges, true, false );
   normalize( hist_base, hist_base, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

}
