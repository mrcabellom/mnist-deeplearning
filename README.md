# Workshop MNIST Deep Learning

## Configuración inicial

1. Instala primero la versión de Anaconda para tu sistema operativo

    * Descargar [aquí](https://repo.continuum.io/archive/Anaconda3-5.0.1-Windows-x86_64.exe) para Windows.

    * Descargar [aquí](https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh) para Linux.

2. Ejecuta el script dentro de scripts
    * environment.bat (si estás en windows)
    * . ./environment.sh (si estás en Linux)

3. Selecciona el nombre que quieres para el entorno virtual (una vez acabe el script tendrás activo el entorno).

4. Ejecuta python main.py para comprobar que cntk se ha instalado correctamente. Dicho programa nos mostrará la version de cntk y las cpu's y gpu's disponibles en nuestra máquina.

## Mnist handwritten database

En la carpeta mnisdata, están disponibles las clases que se encargarán de la lectura/escritura de las imágenes de la base de datos de dígitos escritos a manos.

![alt text](https://www.researchgate.net/profile/Amaury_Lendasse/publication/264273647/figure/fig1/AS:295970354024489@1447576239974/Fig-18-0-9-Sample-digits-of-MNIST-handwritten-digit-database.png)

Los ficheros se descargarán en el directorio especificado en el archivo settings.py. Ambos se serializarán en formato CTF. Habrá un fichero para test y otro para entrenamiento.

La lectura de los archivos se podrá realizar utilizando los deserializadores de CTF disponibles en CNTK.

Para compartir el código en directo, vamos a utilizar la herramienta [codeshare]
