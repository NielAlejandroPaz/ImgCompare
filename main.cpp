#include "imgcompare.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ImgCompare w;
    w.show();
    return a.exec();
}
