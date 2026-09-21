#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <format>
#include <stdexcept>
#include <vector>

#include "activations.hpp"
#include "computational_graph.hpp"
#include "data.hpp"
#include "losses.hpp"
#include "metrics.hpp"
#include "modules.hpp"
#include "ndarray.hpp"
#include "optimizer.hpp"
#include "parameter.hpp"
#include "tensor.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------
// Conversioni NDArray <-> Python. NDArray non viene mai esposta: da qui
// in poi tutto quello che arriva/parte verso Python e' un float, una
// lista o un numpy.ndarray. Sempre copiato, mai vista sulla memoria.
// ---------------------------------------------------------------------
namespace
{

NDArray ndarray_from_pylist(const py::sequence& seq)
{
    if (seq.size() == 0)
        throw std::invalid_argument(
            "impossibile costruire un Tensor da una lista vuota");

    const bool nested = py::isinstance<py::sequence>(seq[0]) &&
                        !py::isinstance<py::str>(seq[0]);

    if (!nested)
    {
        std::vector<float> data;
        data.reserve(seq.size());
        for (const auto& item : seq)
            data.push_back(item.cast<float>());

        return NDArray(data, {data.size()});
    }

    const size_t rows = seq.size();
    const size_t cols = py::cast<py::sequence>(seq[0]).size();

    std::vector<float> data;
    data.reserve(rows * cols);
    for (const auto& item : seq)
    {
        py::sequence row = py::cast<py::sequence>(item);
        if (row.size() != cols)
            throw std::invalid_argument(
                "lista non rettangolare: tutte le righe devono avere la "
                "stessa lunghezza");

        for (const auto& value : row)
            data.push_back(value.cast<float>());
    }

    return NDArray(data, {rows, cols});
}

NDArray ndarray_from_python(const py::object& obj)
{
    // numpy array (qualunque dtype numerico, forzato a float32)
    if (py::isinstance<py::array>(obj))
    {
        auto arr = py::cast<
            py::array_t<float, py::array::c_style | py::array::forcecast>>(obj);

        if (arr.ndim() == 0)
            return NDArray(*arr.data());

        if (arr.ndim() == 1)
        {
            std::vector<float> data(arr.data(), arr.data() + arr.shape(0));
            return NDArray(data, {static_cast<size_t>(arr.shape(0))});
        }

        if (arr.ndim() == 2)
        {
            std::vector<float> data(arr.data(), arr.data() + arr.size());
            return NDArray(data, {static_cast<size_t>(arr.shape(0)),
                                  static_cast<size_t>(arr.shape(1))});
        }

        throw std::invalid_argument("sono supportati solo array numpy 1D o 2D");
    }

    // lista / tupla
    if (py::isinstance<py::sequence>(obj) && !py::isinstance<py::str>(obj))
        return ndarray_from_pylist(py::cast<py::sequence>(obj));

    // scalare (python int/float, numpy scalar, ...)
    try
    {
        return NDArray(obj.cast<float>());
    }
    catch (const py::cast_error&)
    {
        throw std::invalid_argument(
            "Tensor accetta uno scalare, una lista (1D/2D) o un array "
            "numpy");
    }
}

py::array_t<float> ndarray_to_numpy(const NDArray& array)
{
    std::vector<py::ssize_t> shape = {
        static_cast<py::ssize_t>(array.shape()[0]),
        static_cast<py::ssize_t>(array.shape()[1])};

    py::array_t<float> out(shape);
    std::copy(array.data(), array.data() + array.size(), out.mutable_data());

    return out;
}

} // namespace

PYBIND11_MODULE(core, m)
{
    m.doc() = "guess_right: un motore di autodiff in stile PyTorch";

    // -------------------------------------------------------------
    // --- TENSOR ---
    // -------------------------------------------------------------
    py::class_<Tensor> tensor_cls(m, "Tensor");
    tensor_cls
        .def(py::init([](py::object data, bool requires_grad) {
                 if (py::isinstance<Tensor>(data))
                     return data.cast<Tensor>();

                 return Tensor(ndarray_from_python(data), requires_grad);
             }),
             py::arg("data"), py::arg("requires_grad") = false)

        // --- proprieta' ---
        .def_property_readonly("shape",
                               [](const Tensor& self) {
                                   const auto& s = self.shape();
                                   return py::make_tuple(s[0], s[1]);
                               })
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("size", &Tensor::size)
        .def_property_readonly("is_scalar", &Tensor::is_scalar)
        .def_property_readonly("requires_grad", &Tensor::requires_grad)
        .def_property_readonly("grad",
                               [](const Tensor& self) {
                                   return ndarray_to_numpy(self.gradient());
                               })
        .def_property_readonly("T", &Tensor::transpose)

        // --- conversioni verso Python ---
        .def("numpy",
             [](const Tensor& self) { return ndarray_to_numpy(self.item()); })
        .def("item",
             [](const Tensor& self) -> float {
                 if (!self.is_scalar())
                     throw std::runtime_error(
                         "item() funziona solo su tensori scalari, usa "
                         ".numpy() per gli altri casi");

                 return static_cast<float>(self.item());
             })
        .def("__repr__",
             [](const Tensor& self) {
                 return std::format("Tensor({})", self.item());
             })

        // --- autograd ---
        .def("backward", &Tensor::backward)
        .def("zero_grad", &Tensor::zero_grad)
        .def("transpose", &Tensor::transpose)

        // --- operatori ---
        .def("__neg__", [](const Tensor& a) { return -a; })

        .def("__add__", [](const Tensor& a, const Tensor& b) { return a + b; })
        .def("__add__", [](const Tensor& a, float b) { return a + Tensor(b); })
        .def("__radd__", [](const Tensor& a, float b) { return Tensor(b) + a; })

        .def("__sub__", [](const Tensor& a, const Tensor& b) { return a - b; })
        .def("__sub__", [](const Tensor& a, float b) { return a - Tensor(b); })
        .def("__rsub__", [](const Tensor& a, float b) { return Tensor(b) - a; })

        .def("__mul__", [](const Tensor& a, const Tensor& b) { return a * b; })
        .def("__mul__", [](const Tensor& a, float b) { return a * Tensor(b); })
        .def("__rmul__", [](const Tensor& a, float b) { return Tensor(b) * a; })

        .def("__truediv__",
             [](const Tensor& a, const Tensor& b) { return a / b; })
        .def("__truediv__",
             [](const Tensor& a, float b) { return a / Tensor(b); })
        .def("__rtruediv__",
             [](const Tensor& a, float b) { return Tensor(b) / a; })

        .def("__matmul__",
             [](const Tensor& a, const Tensor& b) { return matmul(a, b); })

        .def("__pow__",
             [](const Tensor& a, const Tensor& b) { return pow(a, b); })
        .def("__pow__",
             [](const Tensor& a, float b) { return pow(a, Tensor(b)); })

        .def("__iadd__", &Tensor::operator+=)
        .def("__isub__", &Tensor::operator-=)
        .def("__imul__", &Tensor::operator*=)
        .def("__itruediv__", &Tensor::operator/=)

        // --- costruttori statici ---
        .def_static("zeros", &Tensor::zeros,
                    py::arg("shape") = std::vector<size_t>{},
                    py::arg("requires_grad") = false)
        .def_static("zeros_like", &Tensor::zeros_like, py::arg("other"),
                    py::arg("requires_grad") = false)
        .def_static("ones", &Tensor::ones,
                    py::arg("shape") = std::vector<size_t>{},
                    py::arg("requires_grad") = false)
        .def_static("ones_like", &Tensor::ones_like, py::arg("other"),
                    py::arg("requires_grad") = false)
        .def_static("eye", &Tensor::eye, py::arg("dim"),
                    py::arg("requires_grad") = false)
        .def_static("linspace", &Tensor::linspace, py::arg("start"),
                    py::arg("end"), py::arg("samples") = 50,
                    py::arg("requires_grad") = false)
        .def_static("seed", &Tensor::seed, py::arg("seed"))
        .def_static("normal", &Tensor::normal,
                    py::arg("shape") = std::vector<size_t>{},
                    py::arg("mu") = 0.0f, py::arg("sigma") = 1.0f,
                    py::arg("requires_grad") = false)
        .def_static("uniform", &Tensor::uniform,
                    py::arg("shape") = std::vector<size_t>{},
                    py::arg("a") = 0.0f, py::arg("b") = 1.0f,
                    py::arg("requires_grad") = false);

    // -------------------------------------------------------------
    // --- PARAMETER ---
    // -------------------------------------------------------------
    py::class_<Parameter, Tensor>(m, "Parameter")
        .def(py::init(
                 [](py::object data, bool requires_grad, float weight_decay) {
                     return Parameter(ndarray_from_python(data), requires_grad,
                                      weight_decay);
                 }),
             py::arg("data"), py::arg("requires_grad") = true,
             py::arg("weight_decay") = 1.0f)
        .def_property_readonly("weight_decay", &Parameter::weight_decay);

    // -------------------------------------------------------------
    // --- FUNZIONI ELEMENTWISE (ADL: si chiamano con un Tensor vero) ---
    // -------------------------------------------------------------
    m.def("sum", [](const Tensor& x) { return sum(x); });
    m.def("mean", [](const Tensor& x) { return mean(x); });
    m.def("product", [](const Tensor& x) { return product(x); });
    m.def("matmul",
          [](const Tensor& a, const Tensor& b) { return matmul(a, b); });
    m.def("pow", [](const Tensor& x, const Tensor& e) { return pow(x, e); });
    m.def("pow", [](const Tensor& x, float e) { return pow(x, Tensor(e)); });
    m.def("log", [](const Tensor& x) { return log(x); });
    m.def("exp", [](const Tensor& x) { return exp(x); });
    m.def("sin", [](const Tensor& x) { return sin(x); });
    m.def("cos", [](const Tensor& x) { return cos(x); });
    m.def("tanh", [](const Tensor& x) { return tanh(x); });
    m.def("maximum",
          [](const Tensor& a, const Tensor& b) { return maximum(a, b); });
    m.def("maximum",
          [](float a, const Tensor& b) { return maximum(Tensor(a), b); });
    m.def("maximum",
          [](const Tensor& a, float b) { return maximum(a, Tensor(b)); });
    m.def("minimum",
          [](const Tensor& a, const Tensor& b) { return minimum(a, b); });
    m.def("minimum",
          [](float a, const Tensor& b) { return minimum(Tensor(a), b); });
    m.def("minimum",
          [](const Tensor& a, float b) { return minimum(a, Tensor(b)); });
    m.def("round", [](const Tensor& x) { return round(x); });

    // -------------------------------------------------------------
    // --- MODULE (base) e sottoclassi ---
    // -------------------------------------------------------------
    py::class_<Module, std::shared_ptr<Module>> module_cls(m, "Module");
    module_cls.def("parameters", &Module::parameters)
        .def("forward",
             py::overload_cast<const Tensor&>(&Module::forward, py::const_))
        .def("forward", py::overload_cast<const std::vector<Tensor>&>(
                            &Module::forward, py::const_))
        .def("__call__",
             py::overload_cast<const Tensor&>(&Module::operator(), py::const_))
        .def("__call__", py::overload_cast<const std::vector<Tensor>&>(
                             &Module::operator(), py::const_));

    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<size_t, size_t>(), py::arg("in_features"),
             py::arg("out_features"));

    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(m, "ReLU").def(
        py::init<>());
    py::class_<LeakyReLU, Module, std::shared_ptr<LeakyReLU>>(m, "LeakyReLU")
        .def(py::init<>());
    py::class_<Sigmoid, Module, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>());
    py::class_<Tanh, Module, std::shared_ptr<Tanh>>(m, "Tanh").def(
        py::init<>());
    py::class_<Softmax, Module, std::shared_ptr<Softmax>>(m, "Softmax")
        .def(py::init<>());

    py::class_<Residual, Module, std::shared_ptr<Residual>> residual_cls(
        m, "Residual");
    residual_cls.def(py::init<>())
        .def(
            "add_linear",
            [](Residual& self, size_t in_features, size_t out_features) {
                self.add<Linear>(in_features, out_features);
            },
            py::arg("in_features"), py::arg("out_features"))
        .def("add_relu", [](Residual& self) { self.add<ReLU>(); })
        .def("add_leaky_relu", [](Residual& self) { self.add<LeakyReLU>(); })
        .def("add_sigmoid", [](Residual& self) { self.add<Sigmoid>(); })
        .def("add_tanh", [](Residual& self) { self.add<Tanh>(); })
        .def("add_softmax", [](Residual& self) { self.add<Softmax>(); })
        .def(
            "add_residual",
            [](Residual& self, const Residual& block) {
                self.add<Residual>(block);
            },
            py::arg("block"));

    py::class_<Model, Module, std::shared_ptr<Model>>(m, "Model")
        .def(py::init<>())
        .def(
            "add_linear",
            [](Model& self, size_t in_features, size_t out_features) {
                self.add<Linear>(in_features, out_features);
            },
            py::arg("in_features"), py::arg("out_features"))
        .def("add_relu", [](Model& self) { self.add<ReLU>(); })
        .def("add_leaky_relu", [](Model& self) { self.add<LeakyReLU>(); })
        .def("add_sigmoid", [](Model& self) { self.add<Sigmoid>(); })
        .def("add_tanh", [](Model& self) { self.add<Tanh>(); })
        .def("add_softmax", [](Model& self) { self.add<Softmax>(); })
        .def(
            "add_residual",
            [](Model& self, const Residual& block) {
                self.add<Residual>(block);
            },
            py::arg("block"));

    // -------------------------------------------------------------
    // --- DATASET / LOADER ---
    // -------------------------------------------------------------
    py::class_<Dataset>(m, "Dataset")
        .def(py::init<const std::vector<Tensor>&, const std::vector<Tensor>&>(),
             py::arg("inputs"), py::arg("targets"))
        .def("size", &Dataset::size)
        .def("__len__", &Dataset::size)
        .def_property_readonly("inputs", &Dataset::inputs)
        .def_property_readonly("targets", &Dataset::targets);

    py::class_<Loader>(m, "Loader")
        .def(py::init<const Dataset&, size_t, bool>(), py::arg("dataset"),
             py::arg("batch_size") = 1, py::arg("shuffle") = false,
             py::keep_alive<1, 2>()) // il Dataset deve sopravvivere al Loader
        .def("size", &Loader::size)
        .def("__len__", &Loader::size)
        .def("get", &Loader::get);

    // -------------------------------------------------------------
    // --- OPTIMIZER ---
    // -------------------------------------------------------------
    py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer")
        .def("zero_grad", &Optimizer::zero_grad)
        .def("parameters", &Optimizer::parameters)
        .def("step", &Optimizer::step);

    py::class_<StochasticGradientDescent, Optimizer,
               std::shared_ptr<StochasticGradientDescent>>(
        m, "StochasticGradientDescent")
        .def(py::init<const std::vector<Parameter>&, float>(),
             py::arg("parameters"), py::arg("learning_rate") = 1e-3f);

    // -------------------------------------------------------------
    // --- LOSS ---
    // -------------------------------------------------------------
    py::class_<Loss, std::shared_ptr<Loss>> loss_cls(m, "Loss");
    loss_cls
        .def("__call__",
             py::overload_cast<const Tensor&, const Tensor&>(&Loss::operator()))
        .def("__call__",
             py::overload_cast<const std::vector<Tensor>&,
                               const std::vector<Tensor>&>(&Loss::operator()));

    py::class_<MeanSquaredError, Loss, std::shared_ptr<MeanSquaredError>>(
        m, "MeanSquaredError")
        .def(py::init<>());
    py::class_<BinaryCrossEntropy, Loss, std::shared_ptr<BinaryCrossEntropy>>(
        m, "BinaryCrossEntropy")
        .def(py::init<>());
    py::class_<CrossEntropy, Loss, std::shared_ptr<CrossEntropy>>(
        m, "CrossEntropy")
        .def(py::init<>());

    // -------------------------------------------------------------
    // --- METRICHE ---
    // -------------------------------------------------------------
    m.def(
        "accuracy_score",
        [](const std::vector<Tensor>& predicted,
           const std::vector<Tensor>& target) {
            return static_cast<float>(accuracy_score(predicted, target));
        },
        py::arg("predicted"), py::arg("target"));

    // -------------------------------------------------------------
    // --- NO GRAD (usate solo per costruire il context manager Python) ---
    // -------------------------------------------------------------
    m.def("_set_no_grad", &ComputationalGraph::set_no_grad);
    m.def("_no_grad_enabled", &ComputationalGraph::no_grad);
}
