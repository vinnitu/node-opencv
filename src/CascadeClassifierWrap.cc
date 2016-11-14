#include "CascadeClassifierWrap.h"
#include "OpenCV.h"
#include "Matrix.h"
#include <nan.h>

cv::Point2f operator*(cv::Mat_<double> M, const cv::Point2f& p) {
  cv::Mat_<double> src(3/*rows*/, 1 /*cols*/);

  src(0, 0) = p.x;
  src(1, 0) = p.y;
  src(2, 0) = 1.0;

  cv::Mat_<double> dst = M * src; //USE MATRIX ALGEBRA
  return cv::Point2f(dst(0, 0), dst(1, 0));
}


Nan::Persistent<FunctionTemplate> CascadeClassifierWrap::constructor;

void CascadeClassifierWrap::Init(Local<Object> target) {
  Nan::HandleScope scope;

  Local<FunctionTemplate> ctor = Nan::New<FunctionTemplate> (CascadeClassifierWrap::New);
  constructor.Reset(ctor);
  ctor->InstanceTemplate()->SetInternalFieldCount(1);
  ctor->SetClassName(Nan::New("CascadeClassifier").ToLocalChecked());

  // Prototype
  // Local<ObjectTemplate> proto = constructor->PrototypeTemplate();

  Nan::SetPrototypeMethod(ctor, "detectMultiScale", DetectMultiScale);

  target->Set(Nan::New("CascadeClassifier").ToLocalChecked(), ctor->GetFunction());
}

NAN_METHOD(CascadeClassifierWrap::New) {
  Nan::HandleScope scope;

  if (info.This()->InternalFieldCount() == 0) {
    Nan::ThrowTypeError("Cannot instantiate without new");
  }

  CascadeClassifierWrap *pt = new CascadeClassifierWrap(*info[0]);
  pt->Wrap(info.This());
  info.GetReturnValue().Set( info.This() );
}

CascadeClassifierWrap::CascadeClassifierWrap(v8::Value* fileName) {
  std::string filename;
  filename = std::string(*Nan::Utf8String(fileName->ToString()));

  if (!cc.load(filename.c_str())) {
    Nan::ThrowTypeError("Error loading file");
  }
}

class AsyncDetectMultiScale: public Nan::AsyncWorker {
public:
  AsyncDetectMultiScale(Nan::Callback *callback, CascadeClassifierWrap *cc,
      Matrix* im, double scale, int neighbors, int minw, int minh, double angle, int steps) :
      Nan::AsyncWorker(callback),
      cc(cc),
      im(im),
      scale(scale),
      neighbors(neighbors),
      minw(minw),
      minh(minh),
      angle(angle),
      steps(steps) {
  }
  
  ~AsyncDetectMultiScale() {
  }


  void Rotate2D(double angle, const cv::Mat & src, std::vector < cv::Rect > & g_objects) {
    cv::Point2f center(src.cols/2.0, src.rows/2.0);
    cv::Mat rot = getRotationMatrix2D(center, angle, 1.0);
    cv::Rect bbox = cv::RotatedRect(center, src.size(), angle).boundingRect();
    rot.at<double>(0, 2) += bbox.width/2.0 - center.x;
    rot.at<double>(1, 2) += bbox.height/2.0 - center.y;

    cv::Mat irot;
    invertAffineTransform(rot, irot);

    cv::Mat rotated;
    warpAffine(src, rotated, rot, bbox.size());

    std::vector < cv::Rect > objects;
    this->cc->cc.detectMultiScale(rotated, objects, this->scale, this->neighbors,
        0 | CV_HAAR_SCALE_IMAGE, cv::Size(this->minw, this->minh));

    for (size_t i = 0; i < objects.size(); i++) {
      cv::Rect & r = objects[i];

      cv::Point center(r.x + r.width*.5, r.y + r.height*.5);
      center = irot*center;

      r.x = center.x - r.width*.5;
      r.y = center.y - r.height*.5;

      this->angles.push_back(angle);
      g_objects.push_back(r);
    }
  }

  void Execute() {
    try {
      std::vector < cv::Rect > objects;

      cv::Mat gray;

      if (this->im->mat.channels() != 1) {
        cvtColor(this->im->mat, gray, CV_BGR2GRAY);
        equalizeHist(gray, gray);
      } else {
        gray = this->im->mat;
      }

      this->cc->cc.detectMultiScale(gray, objects, this->scale, this->neighbors,
          0 | CV_HAAR_SCALE_IMAGE, cv::Size(this->minw, this->minh));

      for (unsigned int i = 0; i < objects.size(); i++) {
        this->angles.push_back(0.0);
      }

      if (this->angle) {
        for (int i = 1; i <= this->steps; i++) {
          Rotate2D(-i*this->angle, gray, objects);
          Rotate2D(+i*this->angle, gray, objects);
        }
      }

      res = objects;
    } catch (cv::Exception& e) {
      SetErrorMessage(e.what());
    }
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;
    //  this->matrix->Unref();

    Local < Value > argv[2];
    v8::Local < v8::Array > arr = Nan::New < v8::Array > (this->res.size());

    for (unsigned int i = 0; i < this->res.size(); i++) {
      v8::Local < v8::Object > x = Nan::New<v8::Object>();
      x->Set(Nan::New("x").ToLocalChecked(), Nan::New < Number > (this->res[i].x));
      x->Set(Nan::New("y").ToLocalChecked(), Nan::New < Number > (this->res[i].y));
      x->Set(Nan::New("width").ToLocalChecked(), Nan::New < Number > (this->res[i].width));
      x->Set(Nan::New("height").ToLocalChecked(), Nan::New < Number > (this->res[i].height));
      x->Set(Nan::New("angle").ToLocalChecked(), Nan::New < Number > (this->angles[i]));
      arr->Set(i, x);
    }

    argv[0] = Nan::Null();
    argv[1] = arr;

    Nan::TryCatch try_catch;
    callback->Call(2, argv);
    if (try_catch.HasCaught()) {
      Nan::FatalException(try_catch);
    }
  }

private:
  CascadeClassifierWrap *cc;
  Matrix* im;
  double scale;
  int neighbors;
  int minw;
  int minh;
  double angle;
  int steps;
  std::vector<cv::Rect> res;
  std::vector<double> angles;
};

NAN_METHOD(CascadeClassifierWrap::DetectMultiScale) {
  Nan::HandleScope scope;

  CascadeClassifierWrap *self = Nan::ObjectWrap::Unwrap<CascadeClassifierWrap> (info.This());

  if (info.Length() < 2) {
    Nan::ThrowTypeError("detectMultiScale takes at least 2 info");
  }

  Matrix *im = Nan::ObjectWrap::Unwrap < Matrix > (info[0]->ToObject());
  REQ_FUN_ARG(1, cb);

  double scale = 1.1;
  if (info.Length() > 2 && info[2]->IsNumber()) {
    scale = info[2]->NumberValue();
  }

  int neighbors = 2;
  if (info.Length() > 3 && info[3]->IsInt32()) {
    neighbors = info[3]->IntegerValue();
  }

  int minw = 30;
  int minh = 30;
  if (info.Length() > 5 && info[4]->IsInt32() && info[5]->IsInt32()) {
    minw = info[4]->IntegerValue();
    minh = info[5]->IntegerValue();
  }


  double angle = 0.0;
  if (info.Length() > 6 && info[6]->IsNumber()) {
    angle = info[6]->NumberValue();
  }

  int steps = 1;
  if (info.Length() > 7 && info[7]->IsInt32()) {
    steps = info[7]->IntegerValue();
  }

  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncDetectMultiScale(callback, self, im, scale,
          neighbors, minw, minh, angle, steps));
  return;
}
