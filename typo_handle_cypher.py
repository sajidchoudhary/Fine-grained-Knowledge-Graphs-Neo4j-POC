#Step1
"""CREATE FULLTEXT INDEX kg_nodes_fulltext IF NOT EXISTS
FOR (n:PurchaseOrder|PurchaseRequest|Invoice|Contract)
ON EACH [n.search_text];"""

#Step2
#SHOW INDEXES;

#Step3
'''MATCH (n:PurchaseOrder|PurchaseRequest|Invoice|Contract|KGDocument)
WHERE n.search_text IS NULL
RETURN labels(n) AS label, count(*) AS missing_search_text
ORDER BY missing_search_text DESC;'''
#If you get counts > 0 → you must populate search_text.

#Step4

'''MATCH (n:PurchaseOrder)
SET n.search_text =
  trim(
    coalesce(n.purchase_order_number,"") + " " +
    coalesce(toString(n.purchase_order_version),"") + " " +
    coalesce(n.purchase_order_title,"") + " " +
    coalesce(n.purchase_order_date,"") + " " +
    coalesce(n.event_status,"") + " " +
    coalesce(n.supplier_id,"") + " " +
    coalesce(n.transaction_type,"") + " " +
    coalesce(n.high_risk,"") + " " +
    coalesce(n.payment_terms,"") + " " +
    coalesce(toString(n.total_cost_amount),"") + " " +
    coalesce(n.currency,"") + " " +
    coalesce(n.line_item_description,"") + " " +
    coalesce(n.purchase_requisition_id,"") + " " +
    coalesce(n.invoice_reconciliation_id,"") + " " +
    coalesce(n.lookupKey,"")
  );
'''

'''MATCH (n:Invoice)
SET n.search_text =
  trim(
    coalesce(n.invoice_reconciliation_id,"") + " " +
    coalesce(n.invoice_number,"") + " " +
    coalesce(n.invoice_date,"") + " " +
    coalesce(n.invoice_received_time,"") + " " +
    coalesce(toString(n.gross_amount),"") + " " +
    coalesce(n.invoice_currency,"") + " " +
    coalesce(n.payment_term,"") + " " +
    coalesce(n.invoice_purpose,"") + " " +
    coalesce(n.line_item_description,"") + " " +
    coalesce(n.purchase_order_number,"") + " " +
    coalesce(n.lookupKey,"")
  );
'''

'''MATCH (n:Contract)
SET n.search_text =
  trim(
    coalesce(n.contract_number,"") + " " +
    coalesce(n.agreement_type,"") + " " +
    coalesce(n.agreement_name,"") + " " +
    coalesce(n.agreement_description,"") + " " +
    coalesce(n.contract_start_date,"") + " " +
    coalesce(n.contract_end_date,"") + " " +
    coalesce(n.contract_status,"") + " " +
    coalesce(n.supplier_id,"") + " " +
    coalesce(n.supplier_name,"") + " " +
    coalesce(n.primary_business_unit,"") + " " +
    coalesce(n.user_country,"") + " " +
    coalesce(n.contract_term_type,"") + " " +
    coalesce(n.hierarchy_type,"") + " " +
    coalesce(n.termination_comments,"") + " " +
    coalesce(toString(n.termination_penalty_amount),"") + " " +
    coalesce(n.purchase_order_number,"") + " " +
    coalesce(n.purchase_requisition_id,"") + " " +
    coalesce(n.lookupKey,"")
  );
'''

'''MATCH (n:PurchaseRequest)
SET n.search_text =
  trim(
    coalesce(n.purchase_requisition_id,"") + " " +
    coalesce(n.requester_name,"") + " " +
    coalesce(n.purchase_requisition_title,"") + " " +
    coalesce(n.high_risk,"") + " " +
    coalesce(toString(n.total_cost_amount),"") + " " +
    coalesce(n.currency,"") + " " +
    coalesce(n.responsibility_center,"") + " " +
    coalesce(n.approval_status,"") + " " +
    coalesce(n.approval_date,"") + " " +
    coalesce(n.approver_name,"") + " " +
    coalesce(n.lookupKey,"")
  );
'''