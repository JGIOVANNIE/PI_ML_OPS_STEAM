# _PI_ML_OPS_STEAM_


<img src="https://th.bing.com/th?id=OIP.YoXkq3iMr1JApC1yonem4wHaEK&w=333&h=187&c=8&rs=1&qlt=90&o=6&pid=3.1&rm=2#include" >

este proyecto trabajamos sobre datasets de la plataforma STEAM que es una plataforma de videojuegos, relizando el trabajo de un Data Engineer disponibilzando los dato asi como deployando una API para obtener consultas y realizando un modelo de Machine Learning.
***
<div style="text-align:center;">
    <img src="https://th.bing.com/th/id/OIP.rHVv1cZ1gg3C62zqEuYCUwHaFf?w=234&h=180&c=7&r=0&o=5&pid=1.7" >
</div>

Las consultas que revisremos en este proyecto son las siguientes:

* def PlayTimeGenre( genero : str ): Debe devolver año con mas horas jugadas para dicho género.

* def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.

* *def UsersRecommend( año : int )*: Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)

* def *UsersNotRecommend( año : int )*: Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)

* *def sentiment_analysis( año : int )*: Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.


el sistema de recomndacion user_item:
* *def* *recomendacion_usuario( id de usuario )*: Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.

***

dentro de este repositorio se encuentran todos los archivos utilizados en este proyecto:
* *carpeta API:*
    se enuentran los Dataset reducidos para el consumo en el deploy de la API
* *carpeta Datasets:*
    se encuentran los datasets crudos y los que ya fueron procesados para realizar nuestro ETL Y EDA
* *carpeta Notebooks:* 
    se encuentran los notebooks de cada parte de el proceso: EDA,ETL,ModeloML,Pruebas API
* *Diccionario de Datos Steam* es un diccionario para entender mejor cada una de las columnas en loos Datasets
* *main.py* es el archivo donde se ejecuta nuestra API
***



LinkedIn: www.linkedin.com/in/joshua-giovanni-esquivel-fuentes-06987a190

*******    