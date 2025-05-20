import os
import json
from collections import defaultdict
import glob 
import argparse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredPDFLoader, TextLoader

from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from typing import TypedDict, Annotated, List, Optional, Literal, Union
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate


# Define a Schema for the structured output from model
class Parse_DocFields(BaseModel):
    """Pydantic Class for fields to be extracted"""

    # General fields
    document_id: str = Field(default=None, description="Unique identifier for the document. Generally mentioned at top right of document")
    document_type: Literal["credit", "garnishment", "investment", "personal_account"] = Field(default=None, description="Type of document [credit, garnishment, investment, personal account]")
    document_date: str = Field(default=None, description="Date the document was issued in format: dd.mm.yyyy")
    customer_name: str = Field(default=None, description="Name of the account holder/customer in format: first_name last_name")
    customer_id: int = Field(default=None, description="Customer identification number")
    institution_name: Optional[str] = Field(default=None, description="Name of the financial institution")
    institution_address: Optional[str] = Field(default=None, description="Address of the financial institution in format: street_name street_number, city zipcode, country")
    language: Optional[Literal["en", "de", "fr", "es", "it"]] = Field(default=None, description="Language of the document")

    # Investment fields
    portfolio_id: Optional[str] = Field(default=None, description="Unique identifier for the investment portfolio")
    portfolio_value: Optional[str] = Field(default=None, description="Current total value of all investments in the portfolio (amount currency_symbol)")
    asset_number: Optional[int] = Field(default=None, description="Number of different asset types of the portfolio")
    risk_profile: Optional[str] = Field(default=None, description="Assessment of the portfolio's risk level")

    # Personal account fields
    account_number: Optional[str] = Field(default=None, description="Partially masked bank account number")
    account_type: Optional[str] = Field(default=None, description="Type of account (checking, savings, etc.)")
    statement_period: Optional[str] = Field(default=None, description="Time period covered by the statement")
    opening_balance: Optional[str] = Field(default=None, description="Account balance at the beginning of the statement period (amount currency_symbol)")
    closing_balance: Optional[str] = Field(default=None, description="Account balance at the end of the statement period (amount currency_symbol)")
    available_balance: Optional[str] = Field(default=None, description="Funds available for immediate withdrawal or use (amount currency_symbol)")
    transaction_number: Optional[int] = Field(default=None, description="Number of financial activities affecting the account during the statement period")

    # Garnishment fields
    debtor_name: Optional[str] = Field(default=None, description="Name of the individual whose assets are being garnished")
    creditor_name: Optional[str] = Field(default=None, description="Name of the entity (person/organization) to whom the debt is owed")
    garnishment_amount: Optional[str] = Field(default=None, description="Amount of money to be garnished (amount currency_symbol)")
    effective_date: Optional[str] = Field(default=None, description="Date when the garnishment takes effect (dd.mm.yyyy)")
    duration: Optional[str] = Field(default=None, description="Time period for which the garnishment remains active")
    legal_authority: Optional[str] = Field(default=None, description="Legal basis or jurisdiction authorizing the garnishment")

    # Credit fields
    card_number: Optional[str] = Field(default=None, description="Partially masked credit card number")
    credit_limit: Optional[str] = Field(default=None, description="Maximum amount of credit extended to the customer (amount currency_symbol)")
    interest_rate: Optional[str] = Field(default=None, description="Annual percentage rate applied to outstanding balances (e.g., 5.25%)")
    payment_due_date: Optional[str] = Field(default=None, description="Date by which payment must be received (dd.mm.yyyy)")
    minimum_payment: Optional[str] = Field(default=None, description="Smallest amount that must be paid to maintain account in good standing (amount currency_symbol)")
    previous_balance: Optional[str] = Field(default=None, description="Account balance at the beginning of the credit statement period (amount currency_symbol)")
    new_balance: Optional[str] = Field(default=None, description="Account balance at the end of the credit statement period (amount currency_symbol)")

class PDF_data_Extraction:
    """Class for extracting data from pdf files"""
    def __init__(
            self, 
            chat_model:BaseChatModel, 
            DocFieldSchema:BaseModel, 
            path_fields_json:str
            ):
        
        """Constructor
        Args:
            chat_model (BaseChatModel): LLM chat model
            DocFieldSchema (BaseModel): Pydantic class of structured output schema
            path_fields_json (str): Path of json for fields to be extracted
        """

        # Create attributes
        self.chat_model = chat_model
        self.DocFieldSchema = DocFieldSchema

        # Load extraction fields and dictionary
        with open(path_fields_json, "r") as f:
            self.extraction_fields = json.load(f)

        # TODO Check for same fields in extraction_fields and DocFieldSchema
        # if sum(list(self.extraction_fields.values()), []) != list(DocFieldSchema.model_fields.keys()):
        #     raise ValueError("Fields in extraction_fields.json do not match with DocFieldSchema")

        # update the model with structured output schema
        self.structured_model = self.chat_model.with_structured_output(
            schema=DocFieldSchema,
            method="json_schema",   # support: json_schema, json_mode, function_calling, 
            )

        # Prompt template 
        prompt_message = "Extract the following fields: {fields}. From this document text:\n\n{document}"
        prompt_template = PromptTemplate(
            template=prompt_message,
            input_variables=["fields", "document"],
            validate_template=True
        )

        # Create chains for Document fields and General fields
        self.chain = prompt_template | self.structured_model

    def extract_fields(self, doc_path:str, save_json_path:Union[str, None] = None) -> dict:
        """Extracting the fields from pdf document using LLM
        Args:
            - doc_path (str): Path of pdf file
            - save_json_path (Union[str, None], optional): Path for saving result in json file. Defaults to None.
        Returns:
            - dict: Dictionary contains the extracted fields
        """

        # Load the pdf document in Document class
        loader = PyPDFLoader(file_path=doc_path)
        documents = loader.load()    # List of document class obj

        # Combine all pages
        text = "".join([doc.page_content for doc in documents])

        # Generate the response by invoking the chains
        extracted_fields = self.chain.invoke({
            'fields': list(self.DocFieldSchema.model_fields.keys()),
            'document': text
        })

        # Mapping the extracted_fields to the extraction_fields json
        ret = defaultdict(dict)
        for k, v in self.extraction_fields.items():
            if k == 'general':
                ret['general_field'] = {k:getattr(extracted_fields, k) for k in v}
            else:
                for _k in v:
                    ret['document_field'][_k] = getattr(extracted_fields, _k)

        # Save result in json file
        if save_json_path is not None:
            save_path = os.path.join(save_json_path, f"extracted_results_{os.path.basename(doc_path)}.json")

            # Create folder if it does not exist
            os.makedirs(save_json_path, exist_ok=True)

            # Save extracted_results in json file
            with open(save_path, "w") as f:
                json.dump(ret, f, indent=4)
            
            print(f"[INFO] Extracted fields are saved in file at: {save_path}")

        return ret

if __name__ == "__main__":

    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_path", type=str, default="data/input/doc-04.pdf", help="Path of pdf file")
    parser.add_argument("--save_json_path", type=str, default="data/output", help="Path for saving result in json file")
    
    # Parse arguments
    # args_list = ["--doc_path", "data/input/doc-04.pdf", "--save_json_path", "data/output"]
    # args = parser.parse_args(args_list)
    args = parser.parse_args()

    # Create LLM chat_model
    chat_model = ChatOllama(model="llama3.2:latest")

    # Get the pdf files from the directory
    # pdf_files = sorted(glob.glob(os.path.join('documents', '*pdf')))
    # document path
    # doc_path = os.path.join("documents", "doc-04.pdf")

    # Create class object
    pdf_extractor = PDF_data_Extraction(
        chat_model=chat_model, 
        DocFieldSchema=Parse_DocFields, 
        path_fields_json='data/input/extraction-fields.json'
        )
    
    # Extract data from the pdf and save it in json file 
    extracted_results = pdf_extractor.extract_fields(
        doc_path=args.doc_path,
        save_json_path=args.save_json_path
        )