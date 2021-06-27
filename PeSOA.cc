// Test de Jacobo Casado de Gracia.
// Algorimo PeSOA implementado en C++ por mí. Cogiendo como referencia el PeSOA referenciado en la memoria.
extern "C" {
#include "cec17.h"
}
#include <iostream>
#include <vector>
#include <random>

using namespace std;

int seed = 50;
std::uniform_real_distribution<> dis(-100.0, 100.0);
std::uniform_real_distribution<> sec(0.0, 1.0);
std::mt19937 gen(seed);

int evaluaciones = 0;

void clip(vector<double> &sol, int lower, int upper) {
    for (auto &val : sol) {
        if (val < lower) {
            val = lower;
        }
        else if (val > upper) {
            val = upper;
        }
    }
}

void increm_bias(vector<double> &bias, vector<double> dif) {
    for (unsigned i = 0; i < bias.size(); i++) {
        bias[i] = 0.2*bias[i]+0.4*(dif[i]+bias[i]);
    }
}

void decrement_bias(vector<double> &bias, vector<double> dif) {
    for (unsigned i = 0; i < bias.size(); i++) {
        bias[i] = bias[i]-0.4*(dif[i]+bias[i]);
    }
}

/**
 * Aplica el Solis Wets
 *
 * @param  sol solucion a mejorar.
 * @param fitness fitness de la solución.
 */
template <class Random>
void soliswets(vector<double> &sol, double &fitness, double delta, int maxevals, int lower, int upper, Random &random) {
    const size_t dim = sol.size();
    vector<double> bias (dim), dif (dim), newsol (dim);
    double newfit;
    size_t i;

    int evals = 0;
    int num_success = 0;
    int num_failed = 0;

    while (evals < maxevals) {
        std::uniform_real_distribution<double> distribution(0.0, delta);

        for (i = 0; i < dim; i++) {
            dif[i] = distribution(random);
            newsol[i] = sol[i] + dif[i] + bias[i];
        }

        clip(newsol, lower, upper);
        newfit = cec17_fitness(&newsol[0]);
        evals += 1;
        evaluaciones++;

        if (newfit < fitness) {
            sol = newsol;
            fitness = newfit;
            increm_bias(bias, dif);
            num_success += 1;
            num_failed = 0;
        }
        else if (evals < maxevals) {

            for (i = 0; i < dim; i++) {
                newsol[i] = sol[i] - dif[i] - bias[i];
            }

            clip(newsol, lower, upper);
            newfit = cec17_fitness(&newsol[0]);
            evaluaciones++;
            evals += 1;

            if (newfit < fitness) {
                sol = newsol;
                fitness = newfit;
                decrement_bias(bias, dif);
                num_success += 1;
                num_failed = 0;
            }
            else {
                for (i = 0; i < dim; i++) {
                    bias[i] /= 2;
                }

                num_success = 0;
                num_failed += 1;
            }
        }

        if (num_success >= 5) {
            num_success = 0;
            delta *= 2;
        }
        else if (num_failed >= 3) {
            num_failed = 0;
            delta /= 2;
        }
    }

}

template <typename T>
void remove(std::vector<T>& vec, size_t pos)
{
    typename std::vector<T>::iterator it = vec.begin();
    std::advance(it, pos);
    vec.erase(it);
}

// Como mejor funciona el algoritmo, dice el autor, es con 50 pinguinos por población y alrededor de 5 poblaciones iniciales.

class Pinguino{
public:

    Pinguino(int dim);
    Pinguino();

    void hallarFitness();

    int dim;
    double oxigeno;
    double oxigenoAnterior;
    double tasaVariacionFitness;
    vector<double> posicion;
    vector<double> nuevaPosicion;
    vector<double> variacionFitness;

    double fitness;
    double fitnessAnterior;

    bool necesitaMigrar = false;
    Pinguino& operator=(Pinguino otro);

};

Pinguino::Pinguino(int dim) {

    this->dim = dim;

    for (int i = 0; i < dim; ++i){
        posicion.push_back(dis(gen));
    }

    nuevaPosicion.resize(posicion.size(), 0.0);
    variacionFitness.resize(posicion.size(), 0.0);
    hallarFitness();

    fitnessAnterior = 0.0;
    oxigenoAnterior = 0.0;
    tasaVariacionFitness = 0.0;
    oxigeno = 0.5;

}

Pinguino::Pinguino(){

}

Pinguino& Pinguino::operator=(Pinguino otro) {

    posicion = otro.posicion;
    nuevaPosicion = otro.nuevaPosicion;
    variacionFitness = otro.variacionFitness;
    fitness = otro.fitness;
    oxigeno = otro.oxigeno;
    nuevaPosicion = otro.nuevaPosicion;
    necesitaMigrar = otro.necesitaMigrar;
    dim = otro.dim;
    tasaVariacionFitness = otro.tasaVariacionFitness;

    return *this;
}

void Pinguino::hallarFitness() {
    if (evaluaciones < 100000)
        this->fitness = cec17_fitness(&(posicion[0]));
    evaluaciones++;
}

class poblacionPinguinos{
public:

    poblacionPinguinos(int size, int dim);

    int size;
    double QEF;

    vector<Pinguino> poblacion;
    Pinguino mejorLocal;
    double funcionContribucion;

    Pinguino hallarMejorLocal();

    bool estaVacia();
    void actualizarOxigeno(int i);
    void calcularNuevaPosicion();
    void actualizarPosicionesConMejorFitness();
    void actualizarQEF();
    void recalcularFitness();
    void actualizarFuncionContribucion(double QEFTotal);
    void migrar (poblacionPinguinos &otra, int pos);
    void nadar();
    void BLaMejorLocal();

};

// Parámetros: el tamaño en pinguinos de la poblacion y la dimension por pinguinos.
poblacionPinguinos::poblacionPinguinos(int size, int dim){
    for (int i = 0; i < size; ++i){
        poblacion.push_back(Pinguino(dim));
    }
    this->size = size;
    this->QEF = 0.0;

    recalcularFitness();

    mejorLocal = hallarMejorLocal();

    funcionContribucion = 0.0;

}

void poblacionPinguinos::calcularNuevaPosicion() {

    for (int i = 0; i < poblacion.size(); ++i){
        for (int j = 0; j < poblacion[i].posicion.size(); ++j){
            if (poblacion[i].oxigeno == 0.0){
                poblacion[i].nuevaPosicion[j] = poblacion[i].posicion[j] + (sec(gen) * (mejorLocal.posicion[j] - poblacion[i].posicion[j]));
                if (poblacion[i].nuevaPosicion[j] > 100.00){
                    poblacion[i].nuevaPosicion[j] = dis(gen);
                }
                if (poblacion[i].nuevaPosicion[j] < -100.00){
                    poblacion[i].nuevaPosicion[j] = dis(gen);
                }
            }

            else{
                poblacion[i].nuevaPosicion[j] = poblacion[i].posicion[j] + (poblacion[i].oxigeno * sec(gen) * (mejorLocal.posicion[j] - poblacion[i].posicion[j]));
                if (poblacion[i].nuevaPosicion[j] > 100.00){
                    poblacion[i].nuevaPosicion[j] = dis(gen);
                }
                if (poblacion[i].nuevaPosicion[j] < -100.00){
                    poblacion[i].nuevaPosicion[j] = dis(gen);
                }
            }
        }
    }
}

void poblacionPinguinos::actualizarOxigeno(int i) {

    for (int j = 0; j < poblacion[i].posicion.size(); ++j){

        // cout << "Posicion nueva: " << poblacion[i].nuevaPosicion[j] << ". Posicion antigua: " << poblacion[i].posicion[j] << endl;
        //cout << "Oxigeno inicial : " << poblacion[i].oxigeno << ".  fitness: " <<  (poblacion[i].fitness) << "Fitness nuevo: " << poblacion[i].fitnessNuevaPosicion << ". Diferencia absoluta: " << fabs(poblacion[i].posicion[j] - poblacion[i].nuevaPosicion[j]) << endl;

        poblacion[i].oxigenoAnterior = poblacion[i].oxigeno;
        poblacion[i].oxigeno = poblacion[i].oxigeno + 0.5 * poblacion[i].tasaVariacionFitness;
        //cout << "Oxigeno recalculado: " << poblacion[i].oxigeno << endl;
        // poblacion[i].oxigeno[j] =  poblacion[i].oxigeno[j] / poblacion[i].fitness;

        if (poblacion[i].oxigeno <= 0.0){
            //cout << "NECESITA MIGRAR.";
            poblacion[i].necesitaMigrar = true;
        }
        else
            poblacion[i].necesitaMigrar = false;
        }
}

Pinguino poblacionPinguinos::hallarMejorLocal(){

    int posicionMejor = 0;
    double menorFitness = poblacion[0].fitness;

    for (int i = 0; i < poblacion.size(); ++i){
        if (poblacion[i].fitness < menorFitness){
            posicionMejor = i;
            menorFitness = poblacion[i].fitness;
        }
    }


    mejorLocal = poblacion[posicionMejor];

    return mejorLocal;
}

void poblacionPinguinos::actualizarPosicionesConMejorFitness(){

    for (int i = 0; i < poblacion.size(); ++i){
        for (int j = 0; j < poblacion[i].posicion.size(); ++j){
            if (poblacion[i].variacionFitness[j] > 0.0){
                poblacion[i].posicion[j] = poblacion[i].nuevaPosicion[j];
            }

        }
    }

}

void poblacionPinguinos::actualizarQEF() {

    for (int i = 0; i < poblacion.size(); ++i){
        for (int j = 0; j < poblacion[i].posicion.size(); ++j){
            QEF += (poblacion[i].oxigeno - poblacion[i].oxigenoAnterior);
        }
    }
}

void poblacionPinguinos::recalcularFitness() {
    for (int i = 0; i < poblacion.size(); ++i){
        poblacion[i].hallarFitness();
    }

}

void poblacionPinguinos::actualizarFuncionContribucion(double QEFTotal) {
    funcionContribucion = (QEF / QEFTotal);
}

bool poblacionPinguinos::estaVacia() {
    return (poblacion.size() <= 0);
}

void poblacionPinguinos::migrar(poblacionPinguinos &otra, int pos) {

    Pinguino nuevo = this->poblacion[pos];
    // Lo ponemos a false.
    nuevo.necesitaMigrar = false;
    nuevo.oxigeno = 1.0;
    otra.poblacion.push_back(nuevo);
    otra.recalcularFitness();

    remove(this->poblacion, pos);

}

void poblacionPinguinos::BLaMejorLocal() {
    soliswets(mejorLocal.posicion, mejorLocal.fitness, 10.00, 5000, -100, 100,gen);
}


void poblacionPinguinos::nadar(){

    for (int i = 0; i < poblacion.size(); ++i){

        //cout << "cambio de poblacion" << endl;
        // Muevo a una nueva posicion a el pinguino entero.
        for (int j = 0; j < poblacion[i].posicion.size(); ++j){

            //cout << "cambio de pinguino" << endl;

            poblacion[i].nuevaPosicion[j] = poblacion[i].posicion[j] + (poblacion[i].oxigeno * sec(gen) * (mejorLocal.posicion[j] - poblacion[i].posicion[j]));
            //cout << "Nada de la posicion " << poblacion[i].posicion[j] << " a la posicion " <<  poblacion[i].nuevaPosicion[j] << endl;

            if (poblacion[i].nuevaPosicion[j] > 100.00){
                poblacion[i].nuevaPosicion[j] = dis(gen);
            }
            if (poblacion[i].nuevaPosicion[j] < -100.00){
                poblacion[i].nuevaPosicion[j] = dis(gen);
            }

            // Hallo el fitness de cambiar este elemento en el vector solucion.
            vector<double> posicionesAntiguas = poblacion[i].posicion;
            posicionesAntiguas[j] = poblacion[i].nuevaPosicion[j];
            double fitnessPosterior = cec17_fitness(&(posicionesAntiguas[0]));
            evaluaciones++;

            // Si esta tasa de variacion es positiva, el nuevo fitness es más bajo, por tanto, es mejor solución.
            poblacion[i].variacionFitness[j] = poblacion[i].fitness - fitnessPosterior;

        }

        actualizarPosicionesConMejorFitness();

        // Miro el fitness de la funcion en total.
        poblacion[i].fitnessAnterior = poblacion[i].fitness;
        poblacion[i].fitness = cec17_fitness(&(poblacion[i].posicion[0]));
        evaluaciones++;
        poblacion[i].tasaVariacionFitness = ((poblacion[i].fitnessAnterior  - poblacion[i].fitness));

        //soliswets(poblacion[i].posicion, poblacion[i].fitness, 0.5, 100*30, -100, 100,gen);

        //soliswets(poblacion[i].posicion, poblacion[i].fitness, 1.0, 5000, -100, 100,gen);

        // Si es mejor que el local lo sustiyuyo por este.
        // Además, actualizo su posición.
        if (poblacion[i].fitness > mejorLocal.fitness){
                mejorLocal = poblacion[i];
        }


        actualizarOxigeno(i);


    }

}

// K es el numero de poblaciones (recomendado 5), y N es el numero de pinguinos por poblacion (recomendado 50)
double PeSOA(int numPoblaciones, int numPinguinosporPoblacion, int dim){

    seed = dis(gen);

    vector<poblacionPinguinos> poblaciones;
    Pinguino mejorGlobal;
    int maxEvals = 10000 * dim;

    for (int i = 0; i < numPoblaciones; ++i){
        poblaciones.push_back(poblacionPinguinos(numPinguinosporPoblacion, dim));
        poblaciones[i].hallarMejorLocal();
        //poblaciones[i].BLaMejorLocal();

    }

    while (evaluaciones < maxEvals){

        for (int i = 0; i < numPoblaciones; ++i){
            //poblaciones[i].BLaMejorLocal();
            poblaciones[i].nadar();
            poblaciones[i].actualizarQEF();
            poblaciones[i].hallarMejorLocal();
        }

        // Hallamos el mejor pinguino global. La mejor solucion hasta el momento
        double menorFitness = poblaciones[0].mejorLocal.fitness;
        mejorGlobal = poblaciones[0].mejorLocal;
        double QEFTotal = 0.0;
        for (int i = 0; i < numPoblaciones; ++i){
            //cout << poblaciones[i].mejorLocal.fitness << endl;
            if (poblaciones[i].mejorLocal.fitness < menorFitness){
                mejorGlobal = poblaciones[i].mejorLocal;
                QEFTotal += poblaciones[i].QEF;
            }
        }
        for (int i = 0; i < numPoblaciones; ++i){
            poblaciones[i].actualizarFuncionContribucion(QEFTotal);
        }

        // Si algun pinguino de alguna poblacion necesita migrar, aquí se hace.
        // Se migra en proporción a la función de contribución de cada población.
        for (int i = 0; i < numPoblaciones; ++i){
            for (int j = 0; j < numPoblaciones; ++j){
                if (i != j){
                    for (int k = 0; k < poblaciones[i].poblacion.size(); ++k){
                        if (poblaciones[i].poblacion[k].necesitaMigrar){
                            if (sec(gen) >= poblaciones[j].funcionContribucion){
                                poblaciones[i].migrar(poblaciones[j], k);
                                // En la función migrar, ponemos necesitaMigrar a false.
                            }
                        }
                    }
                }
            }
        }

        // Si el grupo de pinguinos está vacio porque todos han migrado, lo borramos.
        for (int i = 0; i < numPoblaciones; ++i){
            if (poblaciones[i].estaVacia()){
                remove(poblaciones, i);
                numPoblaciones--;
            }
        }

    }

    return cec17_fitness(&(mejorGlobal.posicion[0]));

}

int main() {

    int dim = 10;

    for (int funcid = 1; funcid <= 30; funcid++) {

        evaluaciones = 0;

        cec17_init("PeSOA", funcid, dim);

        //cerr << "Warning: output by console, if you want to create the output file you have to comment cec17_print_output()" << endl;
        //cec17_print_output(); // Comment to generate the output file

        PeSOA(15, 5, dim);

    }

}
