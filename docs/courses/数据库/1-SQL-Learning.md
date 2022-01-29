# SQL-Learning


## Creating the Databases

## The LIKE Operator

Get all the custmers who's patterns start with b/B

```sql
SELECT *
FROM customers
WHERE last_name LIKE 'b%'
```

get all the custmers who's last name start with brush

````sql
SELECT *
FROM customers
WHERE last name LIKE 'brush%'
````

don't care about the first character but the second character is y

```sql
SELECT *
FROM customers
WHERE last name LIKE '_y'
```

- % :  any number of characters
- _  :  single character

EXERCISE

1. Get the customers whose addresses contain TRAIL or AVENUE

```sql
SELECT * 
FROM customers
WHERE addresses LIKE '%trail%' OR 
      addresses LIKE '%avenue%'
```

2. Phone numbers end with 9

```sql
SELECT *
FROM customers
WHERE phone LIKE '%9'
```

<!--more-->

## The REGEXP Operator

REGEXP: regular expression

REGEXP is similar with LIKE

Get all the customers who's last name start with field

```sql
SELECT *
FROM customers
WHERE last_name REGEXP '^field'
```

Get all the customers who's last name end with field

```sql
SLECT * 
FROM customers
WHERE last_name REGEXP 'field$'
```

Get all the customers who have words field,mac or rose in their last name

```sql
SELECT *
FROM customers
WHERE last_name REGEXP 'field|mac|rose'
```

Get all the customers who have words ge,me or ie in their last name

```sql
SELECT *
FROM customers
WHERE last_name REGEXP '[gim]e'
```

- ^ beginning
- $ end
- | logical or
- [abcd]
- [a-f]

EXERCISE 

GET the customers whose

1. first names are ELKA or AMBUR

```sql
SELECT * 
FROM custmoers
WHERE first_name REGEXP 'elka|ambur'
```

2. last names end with EY or ON

```sql
SELECT *
FROM customers
WHERE last_name REGEXP 'ey<img src="https://www.zhihu.com/equation?tex=|on" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">'
```

3. Last names start with MY or contains SE

```sql
SELECT *
FROM customers
WHERE last_name REGEXP '^my|se'
```

4. last name contain B followed by R or U

```sql
SELECT *
FROM customers
WHERE last_name REGEXP 'b[ru]'
```

## The IS NULL Operator

search who does not have a phone

```sql
SELECT *
FROM customers
WHERE phone IS NULL
```

EXERCISE

get the orders that are not shipped

```sql
SEARCH *
FROM orders
WHERE shipped_date IS NULL
```

## The ORDER BY Clause

In this tutor I'm going to show you how to sort data in your sequel queries

Search customers order by first_name

```sql
SELECT *
FROM customers
ORDER BY first_name
```

or you want to reverse the sort order

```sql
SELECT *
FROM customers
ORDER BY first_name DESC
```

sort by multiple columns, for example

```sql
SELECT *
FROM customers
ORDER BY state,first_name;
```

## The LIMIT Caluse

get the first 3 customers

```sql
SELECT *
FROM customers
LIMIT 300;
```

Get customers form 6-9

```sql
SELECT *
FROM customers
LIMIT 6,3;
```

6 is an offset, and 3 means the step

EXERCISE

1. Get the top three loyal customers

```sql
SELECT *
FROM customers
ORDER BY points DESC
LIMIT 3;
```

## The Inner loins

JOIN is equal with INNER JOIN , we don't have to type it.

you can use JOIN to catch the relation between to tables;

For example, you want to search orders which have the same customer_id in table CUSTOMERS from the table ORDERS.

```sql
SELECT *
FROM orders
JOIN customers
    ON orders.customer_id = customers.customer_id
```

we can use alias to simplify the query.

```sql
SELECT *
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
```

if we use o as the orders's alias , we can't write orders to instead o in the next.

### Joining Across Databases

how to combine columns from tables in multiple databases ?

Using database combines with sql_inventory database.

```sql
SELECT *
FROM order_items oi
JOIN sql_inventory.products p
		 ON oi.product_id = p.product_id;
```

the query will be different depending on the database

### Self Joins

In sql we can join tables with itself.

In database sql_hr we have a table named employees.

Now we need select each employee and their manager.

```sql
USE sql_hr;

SELECT *
FROM employees e
JOIN employees m
     On e.reports_to = m.employee_id
```

### Joining Multiple Tables

```sql
SELECT 
		o.order_id,
		o.order_data,
		c.first_name,
		c.last_name,
		os.name AS status
FROM order o
JOIN customers c
		ON o.customers_id = c.customer_id
JOIN order_statuses os
		ON o.status = os.order_status_id
```

**Exercise**

In database sql_invoicing we have this table, **payments** and these are the payments that each client has made towards either invoice. We  also have a table named **payment_methods**.

Write a query that join the  payments with the **payment methods** tables as well as the clients table.Produce a report that shows the payments, with more details, such as the name of the client and the payment method.

```sql
SELECT 
		p.data,
		p.invoice_id,
		p.amount,
		c.name,
		pm.name,
FROM payments p
JOIN clients c
		ON p.client_id = c.client_id
JOIN payment_methods pm
		ON p.payment_method = pm.payment_method_id
```

### Compound Join Conditions

we have mutiple conditions to join these two tables

e.g.

```sql
SELECT * 
FROM order_items oi
JOIN oder_item_notes oin
		ON oi.order_id = oin.order_id
		AND oi.product_id = oin.product_id;
```

### Implicit Join Syntax

In Mysql we can use simple query to instead Join condition.

for example

```sql
SELECT *
FROM orders o
JOIN customers c
		ON o.customer_id = c.customer_id
		
-- Implpicit Join Syntax
SELECT *
FROM orders o,customers c
WHERE o.customer_id = c.customer_id
```















